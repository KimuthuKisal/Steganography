import torch
from PIL import Image
import os
import config
from typing import Callable
from torch.utils.data import Dataset
import numpy as np

class SourceTargetDataset(Dataset):
    def __init__(self, root_source:str, root_target:str, transform:Callable=None, resize_to: tuple=(256,256)):
        self.root_source = root_source
        self.root_target = root_target
        self.transform = transform
        self.resize_to = resize_to
        self.root_source_images = os.listdir(root_source)
        self.root_target_images = os.listdir(root_target)
        self.length_source = len(self.root_source_images)
        self.length_target = len(self.root_target_images)
        self.length_dataset = max(len(self.root_source_images), len(self.root_target_images))
    def __len__(self):
        return self.length_dataset
    def __getitem__(self, index:int):
        source_image = self.root_source_images[index % self.length_source]
        target_image = self.root_target_images[index % self.length_target]
        source_path = os.path.join(self.root_source, source_image)
        target_path = os.path.join(self.root_target, target_image)
        source_image_np = np.array(Image.open(source_path).convert("RGB").resize(self.resize_to))
        target_image_np = np.array(Image.open(target_path).convert("RGB").resize(self.resize_to))
        if self.transform:
            augmentations = self.transform(image=source_image_np, image0=target_image_np)
            source_image_np = augmentations["image"]
            target_image_np = augmentations["image0"]
        return source_image_np, target_image_np