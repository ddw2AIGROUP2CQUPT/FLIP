import numpy as np
import time
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
from models import utils

@torch.no_grad()
def evaluation(args, model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    # ======================================== text feature ======================================== #
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35,
                                     return_tensors="pt").to(device)
        # token = model.tokenizer._tokenize(text[0])
        # input_id = [model.tokenizer._convert_token_to_id(t) for t in token]
        # _token = [model.tokenizer._convert_id_to_token(idx) for idx in input_id]
        # print(token == _token)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    # ======================================== image&sketch feature ======================================== #
    image_feats = []
    image_embeds = []
    sketch_sde_feats = []
    sketch_sde_embeds = []
    for i, (image, sketch_sde, img_id) in enumerate(data_loader): 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)  

        sketch_sde = sketch_sde.view(-1, image.shape[1], image.shape[2], image.shape[3]).to(device) 
        sketch_sde_feat = model.visual_encoder(sketch_sde)
        sketch_sde_embed = model.vision_proj(sketch_sde_feat[:,0,:])#.view(image_embed.shape[0], -1, image_embed.shape[1])
        sketch_sde_embed = F.normalize(sketch_sde_embed,dim=-1)
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
        sketch_sde_feats.append(sketch_sde_feat.cpu())
        sketch_sde_embeds.append(sketch_sde_embed)
     
    image_feats = torch.cat(image_feats,dim=0).to(device)
    image_embeds = torch.cat(image_embeds,dim=0).to(device)
    sketch_sde_feats = torch.cat(sketch_sde_feats,dim=0).to(device)#.view(image_feats.shape[0], -1, image_feats.shape[1], image_feats.shape[2] )
    sketch_sde_embeds = torch.cat(sketch_sde_embeds,dim=0).to(device)

    # for i in range(len(text_embeds)):
    #     sketch_sde_embeds[i*50:(i+1)*50] = sketch_sde_embeds[i*50:(i+1)*50] + text_embeds[i]

    sketch_sde_embeds = F.normalize(sketch_sde_embeds, dim=-1)

    print('Computing features Cost time {}'.format(time.time() - start_time))

    # ======================================== i2t score ======================================== #
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    # ======================================== t2i score ======================================== #    
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                    attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    # # ======================================== s2i score ed distance ======================================== #
    # score_matrix_s2i = torch.zeros(len(sketch_sde_embeds), 1).to(device)
    # step = sketch_sde_embeds.size(0) // num_tasks + 1
    # start = rank * step
    # end = min(sketch_sde_embeds.size(0), start + step)

    # for i, sketch_sde_embed in enumerate(metric_logger.log_every(sketch_sde_embeds[start:end], 50, header)):
    #     target_distance = F.pairwise_distance(sketch_sde_embed, image_embeds[(start + i) // 50])
    #     distance = F.pairwise_distance(sketch_sde_embed.unsqueeze(0), image_embeds)
    #     rank_sim = distance.le(target_distance).sum()
    #     if rank_sim == 0:
    #         score_matrix_s2i[start + i, :] = 1
    #     else:
    #         score_matrix_s2i[start + i, :] = rank_sim

    # score_matrix_s2i = score_matrix_s2i.view(-1, 50)

    # ======================================== s2i score cos sim ======================================== #
    score_matrix_s2i = torch.zeros(len(sketch_sde_embeds), 1).to(device)
    step = sketch_sde_embeds.size(0) // num_tasks + 1
    start = rank * step
    end = min(sketch_sde_embeds.size(0), start + step)

    for i, sketch_sde_embed in enumerate(metric_logger.log_every(sketch_sde_embeds[start:end], 2500, header)):
        sketch_sde_embed = sketch_sde_embed.unsqueeze(0)# + text_embeds[(start + i) // 50].unsqueeze(0)
        cosine_similarity = F.cosine_similarity(sketch_sde_embed, image_embeds, dim=1)
        rank_sim = (cosine_similarity >= cosine_similarity[(start + i) // 50]).sum().item()  # 计算排名
        if rank_sim == 0:
            score_matrix_s2i[start + i, :] = 1
        else:
            score_matrix_s2i[start + i, :] = rank_sim

    score_matrix_s2i = score_matrix_s2i.view(-1, 50)


    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_s2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_s2i.cpu().numpy(),  score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_s2i, scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    # Sketch -> Images
    num_sample, num_step = len(scores_s2i), len(scores_s2i[0])
    rank_s2i_percentile = (num_sample - scores_s2i) / (num_sample - 1)
    exps = np.linspace(1, num_step, num_step) / num_step
    factor = np.exp(1 - exps) / np.e

    mb = []
    ma = []
    wmb = []
    wma = []
    for i in range(num_sample):
        rank = scores_s2i[i]
        rank_percentile = rank_s2i_percentile[i]
        mb.append(np.sum(1 / rank) / len(rank))
        ma.append(np.sum(rank_percentile) / len(rank_percentile))
        wmb.append(np.sum((1 / rank) * factor) / len(rank))
        wma.append(np.sum(rank_percentile * factor) / len(rank_percentile))

    mb = np.mean(mb) * 100
    ma = np.mean(ma) * 100
    wmb = np.mean(wmb) * 100
    wma = np.mean(wma) * 100

    eval_result = {'MB': mb,
                   'MA': ma,
                   'WMB': wmb,
                   'WMA': wma,
                   'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    
    # eval_result = {
    #                'txt_r1': tr1,
    #                'txt_r5': tr5,
    #                'txt_r10': tr10,
    #                'txt_r_mean': tr_mean,
    #                'img_r1': ir1,
    #                'img_r5': ir5,
    #                'img_r10': ir10,
    #                'img_r_mean': ir_mean,
    #                'r_mean': r_mean}
    return eval_result