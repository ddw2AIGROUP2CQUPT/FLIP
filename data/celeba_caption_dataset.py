import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from data.utils import pre_caption
import random


class celeba_caption_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=65, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filename = 'CelebA_Caption_train.json'

        # download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if not os.path.exists(os.path.join(self.image_root, ann['image'])):
                print(os.path.join(self.image_root, ann['image']))
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB') 
        image = self.transform(image)
        caption = self.prompt + pre_caption(*ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    
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
            if not os.path.exists(os.path.join(self.image_root, ann['image'])):
                print(os.path.join(self.image_root, ann['image']))
                
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