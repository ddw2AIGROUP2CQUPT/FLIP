import numpy as np
import time
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
from models import utils


def label_wise_accuracy(output, target):
    """
    :param output: model's output (before sigmoid), shape: [batch_size, num_labels]
    :param target: ground truth labels, shape: [batch_size, num_labels]
    :return: label-wise averaged accuracy
    """
    output = torch.sigmoid(output) > 0.5
    corrects = (output == target).sum(dim=0).float()
    accuracy_per_label = corrects / target.size(0)
    return accuracy_per_label.mean().item()

def sample_wise_accuracy(output, target):
    """
    :param output: model's output (before sigmoid), shape: [batch_size, num_labels]
    :param target: ground truth labels, shape: [batch_size, num_labels]
    :return: sample-wise averaged accuracy
    """
    output = torch.sigmoid(output) > 0.5
    correct_samples = (output == target).all(dim=1).float()
    return correct_samples.mean().item()


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
    
    pred_logits = []
    targets = []
    for i, (image, target) in enumerate(data_loader):
        image = image.to(device)
        target = target.to(device)

        image_output = model.visual_encoder(image, intermediate_hidden_state=model.intermediate_hidden_state)
        layer_output = image_output.intermediate_hidden_state
        layer_output['layer_11'] = image_output.last_hidden_state
        
        layer_embeds_list = []
        for output in layer_output.values():
            first_token = model.ln(output[:,0,:]).unsqueeze(1)
            second_token = model.ln(torch.mean(output[:, 1:, :], dim=1)).unsqueeze(1)
            third_token = model.ln(torch.max(output[:, 1:, :], dim=1)[0]).unsqueeze(1)
            layer_embeds_list.extend([first_token, second_token, third_token])
        layer_embeds = torch.cat(layer_embeds_list, dim=1)
        
        combined_embeds = torch.sum(model.layer_weights * layer_embeds, dim=1)
        pred_logits.extend(model.visual_proj(combined_embeds))
        targets.extend(target)

    pred_logits = torch.stack(pred_logits).to(model.visual_encoder.device)
    targets = torch.stack(targets).to(model.visual_encoder.device)
    return pred_logits, targets


@torch.no_grad()
def eval(pred, target):
    acc = label_wise_accuracy(pred, target)
    eval_result = {'acc': acc}
    return eval_result
    
