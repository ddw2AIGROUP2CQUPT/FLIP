'''
@author yutangli
'''
import transformers
transformers.logging.set_verbosity_error()

from models.fflip import (
    VisionConfig, 
    VisionModel,
    BertModel, 
    BertConfig,
    MMSEG_UPerHead,
    init_tokenizer,
    load_checkpoint,
    _make_fpns)
from .utils import heatmap2points, points2heatmap, visualize_in_row, denormalize_points, normalize_points, resize_embedding
import torch
from torch import nn
import torch.nn.functional as F


class Aflw19_Alignment(nn.Module):
    def __init__(self,                 
                 visual_config = '/home/ubuntu/lxd-workplace/LYT/FFLIP/itc_itm_mm/configs/vision_config.json',
                 vit = 'base',
                 image_size = 224,
                 intermediate_hidden_state=True,
                 num_landmarks=19,
                 flags = None,
                 heatmap_size = 128,
                 heatmap_radius = 5.0,
                 heatmap_interpolate_mode = 'bilinear',
                 loss_weights = {'coord_l1_loss': 1.0, 'heatmap_ce_loss': 1.0}
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.intermediate_hidden_state = intermediate_hidden_state
        self.flags = flags
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_radius = heatmap_radius
        self.heatmap_interpolate_mode = heatmap_interpolate_mode
        self.loss_weights = loss_weights

        if vit == 'base':
            self.vision_config = VisionConfig().from_json_file(visual_config)
            self.visual_encoder = VisionModel.from_pretrained("openai/clip-vit-base-patch16", config = self.vision_config)
        elif vit == 'large':
            self.vision_config = VisionConfig().from_json_file(visual_config)
            self.visual_encoder = VisionModel.from_pretrained("openai/clip-vit-large-patch14", config = self.vision_config)

        if self.image_size is not None and \
                self.vision_config.image_size != self.image_size:
            # resizing the positonal embeddings
            self.visual_encoder.vision_model.embeddings.position_embedding = resize_embedding(
                self.visual_encoder.vision_model.embeddings.position_embedding, self.image_size // self.vision_config.patch_size)
            self.visual_encoder.vision_model.embeddings.num_patches = (self.image_size // self.vision_config.patch_size) ** 2
            self.visual_encoder.vision_model.embeddings.num_positions = (self.image_size // self.vision_config.patch_size) ** 2 + 1
            self.visual_encoder.vision_model.embeddings.position_ids = torch.arange(self.visual_encoder.vision_model.embeddings.num_positions).expand((1, -1))

        self.vision_width = self.visual_encoder.config.hidden_size
        self.patch_num = self.image_size // self.vision_config.patch_size

        self.fpns = _make_fpns(self.vision_config.patch_size, self.vision_width)
        self.heatmap_head = MMSEG_UPerHead(num_classes=num_landmarks, in_channels=[self.visual_encoder.get_output_channel(vit)]*4, channels=self.vision_width)

        self.register_buffer('image_mean', torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))


    def forward(self, data):
        # b x c x h x w
        image = data['warped_image'].cuda().permute(0, 3, 1, 2).contiguous()
        _, _, h, w = image.shape
        if h != self.image_size or w != self.image_size:
            image = F.interpolate(image, self.image_size,
                                  mode='bilinear', align_corners=False)
        image = (image - self.image_mean) / self.image_std

        with torch.no_grad():
            layer_output = {}
            image_output = self.visual_encoder(image, intermediate_hidden_state=self.intermediate_hidden_state)
            layer_output = image_output.intermediate_hidden_state
            layer_output['layer_11'] = image_output.last_hidden_state

        features = [output[:, 1:, :].permute(0, 2, 1).reshape(-1, self.vision_width, self.patch_num, self.patch_num)
                     for output in layer_output.values()]
        cls_tokens = [output[:, 0, :] for output in layer_output.values()]

        if self.fpns is not None:
            for i, fpn in enumerate(self.fpns):
                features[i] = fpn(features[i])
        
        heatmap = self.heatmap_head(features)
        heatmap_acted = F.sigmoid(heatmap)
        pred_landmark = heatmap2points(heatmap_acted)
        aux_outputs = {'heatmap': heatmap, 'heatmap_acted': heatmap_acted}

        cache = dict()
        if self.flags['with_losses']:
            landmark = normalize_points(
                data['warped_landmarks'].to(image), h, w)

            # compute all losses
            def _compute_named_loss(name: str) -> torch.Tensor:
                if name == 'coord_l1_loss':
                    return (landmark - pred_landmark).norm(dim=-1).mean([1])

                if name.startswith('heatmap'):
                    if 'pred_heatmap' not in cache:
                        cache['pred_heatmap'] = F.interpolate(
                            aux_outputs['heatmap'], (self.heatmap_size,
                                                     self.heatmap_size),
                            mode=self.heatmap_interpolate_mode, align_corners=False)
                    if 'pred_heatmap_acted' not in cache:
                        cache['pred_heatmap_acted'] = F.interpolate(
                            aux_outputs['heatmap_acted'], (self.heatmap_size,
                                                           self.heatmap_size),
                            mode=self.heatmap_interpolate_mode, align_corners=False)
                    if 'heatmap' not in cache:
                        # render gt heatmap
                        with torch.no_grad():
                            cache['heatmap'] = points2heatmap(
                                landmark, (self.heatmap_size, self.heatmap_size), self.heatmap_radius)

                if name == 'heatmap_l1_loss':
                    return (cache['pred_heatmap_acted'] - cache['heatmap']).abs().mean([1, 2, 3])
                if name == 'heatmap_l2_loss':
                    return (cache['pred_heatmap'] - cache['heatmap']).pow(2).mean([1, 2, 3])
                if name == 'heatmap_ce_loss':
                    bce_loss = F.binary_cross_entropy_with_logits(
                        cache['pred_heatmap'], cache['heatmap'], reduction='none')
                    return bce_loss.mean([1, 2, 3])

                raise RuntimeError(f'Unknown loss name: {name}.')

            losses = {name: _compute_named_loss(
                name) for name, weight in self.loss_weights.items() if weight != 0.0}
            loss = sum([l * self.loss_weights[name]
                        for name, l in losses.items()]).mean()
        else:
            loss, losses = None, dict()

        if self.flags['with_outputs']:
            outputs = {'pred_warped_landmarks': denormalize_points(
                pred_landmark, h, w)}
            if 'heatmap' in cache:
                outputs['heatmap'] = cache['heatmap']
            if 'pred_heatmap' in cache:
                outputs['pred_heatmap'] = cache['pred_heatmap']
            if 'pred_heatmap_acted' in cache:
                outputs['pred_heatmap_acted'] = cache['pred_heatmap_acted']
        else:
            outputs = dict()

        if self.flags['with_images']:
            images = {
                'pred_warped_landmarks': visualize_in_row(((pred_landmark, image), 'points'))}
            if 'heatmap' in cache:
                images['heatmap'] = visualize_in_row(
                    (cache['heatmap'], 'BNHW'))
                images['heatmap_sum'] = visualize_in_row(
                    (cache['heatmap'].sum(1), 'BHW'))

            if 'pred_heatmap_acted' in cache:
                images['pred_heatmap_acted'] = visualize_in_row(
                    (cache['pred_heatmap_acted'], 'BNHW'))
                images['pred_heatmap_acted_sum'] = visualize_in_row(
                    (cache['pred_heatmap_acted'].sum(1), 'BHW'))
        else:
            images = dict()

        return loss, losses, outputs, images



def aflw19_alignment(pretrained='', **kwargs):
    model = Aflw19_Alignment(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model 

        
