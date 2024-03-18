import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import pandas as pd
import numpy as np


class lfwa_train(Dataset):
    def __init__(self, transform, image_root, csv_root):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        csv_root (string): directory to store the csv file
        """
        csv_path = 'list_attr_lfwa.csv'
        self.csv = pd.read_csv(os.path.join(csv_root, csv_path)).values
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        sample = self.csv[idx]
        target = np.array(sample[1:]).astype(np.float32)
        image_path = os.path.join(self.image_root, sample[0])
        image = Image.open(image_path)
        image = self.transform(image)
        return image, target


class lfwa_test(Dataset):
    def __init__(self, transform, image_root, csv_root):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        csv_root (string): directory to store the csv file
        """
        csv_path = 'list_attr_lfwa_test.csv'
        self.csv = pd.read_csv(os.path.join(csv_root, csv_path)).values
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        sample = self.csv[idx]
        target = np.array(sample[1:]).astype(np.float32)
        image_path = os.path.join(self.image_root,sample[0])
        image = Image.open(image_path)
        image = self.transform(image)
        return image, target