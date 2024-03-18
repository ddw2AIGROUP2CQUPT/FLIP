"""
@author yutangli
"""
import os
import gradio as gr
from gradio import components as gc
from models.fflip_celeba_caption_retrieval import celeba_caption_retrieval
from data.utils import pre_caption
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import torch
import time


## parameter
image_size = 224
image_root = '/home/ubuntu/lxd-workplace/LYT/FFLIP/CelebA/images'
ann_root = '/home/ubuntu/lxd-workplace/LYT/FFLIP/CelebA/annotation'
model_path = '/home/ubuntu/lxd-workplace/LYT/FFLIP/itc_itm_mm/outputs/celeba_caption_retrieval/checkpoint_best.pth'
k_test = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataset(image_size, image_root, ann_root):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    test_dataset = celeba_caption_test(transform_test, image_root, ann_root, 'test')
    return test_dataset


class celeba_caption_test(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=65):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        filenames = {'test': 'CelebA_Caption_test.json',
                     'val': 'CelebA_Caption_eval.json'}
        
        # download_url(urls[split], ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        return image, index


dataset = create_dataset(image_size, image_root, ann_root)
data_loader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True, shuffle=False)
model = celeba_caption_retrieval(pretrained=model_path, vit='base', queue_size=61440).to(device)

with torch.no_grad():
    start_time = time.time()
    # ======================================== text feature ======================================== #
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 16
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=65,
                                        return_tensors="pt").to(device)
        text_feat = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_feat.last_hidden_state[:,0,:]), dim=-1)
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0).cpu()
    text_ids = torch.cat(text_ids, dim=0).cpu()
    text_atts = torch.cat(text_atts, dim=0).cpu()

    # ======================================== image&sketch feature ======================================== #
    image_feats = []
    image_embeds = []
    for i, (image, img_id) in enumerate(data_loader): 
        image = image.cuda() 
        image_feat = model.visual_encoder(image).last_hidden_state
        image_embed = F.normalize(model.vision_proj(image_feat[:,0,:]), dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0).cpu()
    image_embeds = torch.cat(image_embeds, dim=0).cpu()
    print('Computing features Cost time {}'.format(time.time() - start_time))


# 文本到图像的检索函数
def text_to_face(text_input):
    text_input = pre_caption(text_input, max_words=65)
    text_input = model.tokenizer(text_input, padding='max_length', truncation=True, max_length=65,
                                    return_tensors="pt").to(device)
    text_feat = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_feat.last_hidden_state[:,0,:]), dim=-1).cpu()

    sims_matrix = text_embed @ image_embeds.t()

    topk_sim, topk_idx = sims_matrix.topk(k_test, dim=1)
    topk_idx = topk_idx.squeeze(0).numpy()
    result_image = [os.path.join(image_root, data_loader.dataset.image[topk_idx[i]]) for i in range(len(topk_idx))]
    idxs_str = np.array([data_loader.dataset.image[topk_idx[i]] for i in range(len(topk_idx))])
    idxs_str = np.array2string(idxs_str, separator=', ', formatter={'all': lambda x: f'"{x}"'})
    return result_image, idxs_str # 返回图像和空文本

# 图像到文本的检索函数
def face_to_text(image_input):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    image = transform_test(Image.fromarray(image_input))
    image = image.unsqueeze(0).to(device)
    image_feat = model.visual_encoder(image).last_hidden_state
    image_embed = F.normalize(model.vision_proj(image_feat[:,0,:]), dim=-1).cpu()

    sims_matrix = image_embed @ text_embeds.t()
    topk_sim, topk_idx = sims_matrix.topk(k_test, dim=1)
    topk_idx = topk_idx.squeeze(0).numpy()
    result_text = np.array([data_loader.dataset.text[topk_idx[i]] for i in range(len(topk_idx))])
    result_text = np.array2string(result_text, separator=', ', formatter={'all': lambda x: f'"{x}"'})
    return None, result_text  # 返回空图像和文本

# 合并函数
def combined_function(text_input=None, image_input=None):
    if text_input and text_input.strip():  # 检查文本输入是否非空
        return text_to_face(text_input)
    elif image_input is not None and image_input.size > 0:  # 检查图像输入是否存在且非空
        return face_to_text(image_input)
    else:
        return None, "No valid input provided"  # 提供一个默认输出

# 定义Gradio界面
iface = gr.Interface(
    fn=combined_function,
    inputs=[
        gc.Textbox(lines=5, label="文本输入", default=""),
        gc.Image(label="图像输入")
    ],
    outputs=[
        gc.Gallery(label="检索到的图像"),
        gc.Textbox(lines=5, label="检索到的文本描述")
    ]
)

# 运行Gradio界面
iface.launch(server_name="172.20.5.8", share=True)
