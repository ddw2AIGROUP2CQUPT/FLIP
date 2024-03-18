'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from models.fflip_celeba_recognition import celeba_recognition
import utils
from eval.celeba_recognition import evaluation, eval
from utils import warmup_lr_schedule, step_lr_schedule, cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('ce_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   

    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        optimizer.zero_grad()

        ce_loss = model(image, target)

        ce_loss.backward()
        optimizer.step()

        metric_logger.update(ce_loss=ce_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    train_dataset, test_dataset = create_dataset(config['dataset'], config)
    # 随机抽样获取10%的训练数据
    train_data_length = int(len(train_dataset) * 0.1)
    random.seed(42)  # 设置随机种子以确保可重复性
    random_indices = random.sample(range(len(train_dataset)), train_data_length)
    train_dataset = [train_dataset[i] for i in random_indices]
    # 随机抽样获取1%的训练数据
    # train_data_length = int(len(train_dataset) * 0.01)
    # random.seed(42)  # 设置随机种子以确保可重复性
    # random_indices = random.sample(range(len(train_dataset)), train_data_length)
    # train_dataset = [train_dataset[i] for i in random_indices]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                            batch_size=[config['batch_size_train']] + [config['batch_size_test']],
                                            num_workers=[8, 8],
                                            is_trains=[True, False],
                                            collate_fns=[None, None])
    #### Model #### 
    print("Creating model")
    model = celeba_recognition(pretrained=config['pretrained'], vit=config['vit'], num_classes=config['num_classes'], intermediate_hidden_state=config['intermediate_hidden_state'])
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] 
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module    
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                    train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                    
            train_stats = train(model, train_loader, optimizer, epoch, device, config) 

        pred, target = evaluation(args, model_without_ddp, test_loader, device, config)
        
        if utils.is_main_process():  
            test_result = eval(pred, target)
            print(test_result)
            if args.evaluate:
                log_stats = {**{f'test_{k}': v for k, v in test_result.items()}}
                with open(os.path.join(args.output_dir, "evaluate_log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }
                with open(os.path.join(args.output_dir, "train_log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")     

            save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

            if test_result['acc'] > best:
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = test_result['acc']
                best_epoch = epoch   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/ubuntu/lxd-workplace/LYT/FFLIP/itc_itm_mm/configs/lfwa_recognition.yaml')
    parser.add_argument('--output_dir', default='output/lfwa_recognition_10_F2')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool, help='whether to use distributed mode to training')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)