import transformers
transformers.logging.set_verbosity_error()

from models.fflip import (
    VisionConfig, 
    VisionModel,
    BertModel, 
    BertConfig,
    init_tokenizer,
    load_checkpoint)

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class Celeba_Recognition(nn.Module):
    def __init__(self,
                 vit = 'base',
                 num_classes = 40,
                 intermediate_hidden_state=False
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.intermediate_hidden_state = intermediate_hidden_state

        if vit == 'base':
            self.vision_config = VisionConfig().from_json_file('/home/ubuntu/lxd-workplace/LYT/FFLIP/itc_itm_mm/configs/vision_config.json')
            self.visual_encoder = VisionModel.from_pretrained("openai/clip-vit-base-patch16", config = self.vision_config)
            vision_width = self.visual_encoder.config.hidden_size
        intermediate_num = len(self.visual_encoder.config.intermediate_transformer_output)+1
        self.layer_weights = nn.Parameter(torch.randn(1, 3*intermediate_num, 1)).to(self.visual_encoder.device)

        self.ln = nn.LayerNorm(vision_width, eps=1e-05)
        self.visual_proj = nn.Linear(vision_width, num_classes)
        self.ce_loss = nn.BCEWithLogitsLoss()

        # 初始化
        nn.init.xavier_uniform_(self.layer_weights)
        nn.init.xavier_uniform_(self.visual_proj.weight)
        

    def forward(self, image, target):
        self.train()
        layer_output = {}
        with torch.no_grad():
            image_output = self.visual_encoder(image, intermediate_hidden_state=self.intermediate_hidden_state)
            layer_output = image_output.intermediate_hidden_state
            layer_output['layer_11'] = image_output.last_hidden_state
        
        layer_embeds_list = []
        for output in layer_output.values():
            first_token = self.ln(output[:,0,:]).unsqueeze(1)
            second_token = self.ln(torch.mean(output[:, 1:, :], dim=1)).unsqueeze(1)
            third_token = self.ln(torch.max(output[:, 1:, :], dim=1)[0]).unsqueeze(1)
            layer_embeds_list.extend([first_token, second_token, third_token])
        layer_embeds = torch.cat(layer_embeds_list, dim=1)
        
        combined_embeds = torch.sum(self.layer_weights * layer_embeds, dim=1)
        pred_logits = self.visual_proj(combined_embeds)
        target = target.to(pred_logits.device)
        ce_loss = self.ce_loss(pred_logits, target)

        return ce_loss


def celeba_recognition(pretrained='', **kwargs):
    model = Celeba_Recognition(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model 
