import torch
from PIL import Image
import os
import config
from typing import Callable, Tuple
from torch.utils.data import Dataset, DataLoader  
import numpy as np
import random
from torchvision import transforms


# Class to pass single image
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
        # print("Indexes :", index % self.length_source, " and ", index % self.length_target)
        source_image = self.root_source_images[index % self.length_source]
        target_image = self.root_target_images[index % self.length_target]
        source_path = os.path.join(self.root_source, source_image)
        target_path = os.path.join(self.root_target, target_image)
        source_image_np = np.array(Image.open(source_path).convert("RGB").resize(self.resize_to))
        target_image_np = np.array(Image.open(target_path).convert("RGB").resize(self.resize_to))
        if self.transform:
            augmentations = self.transform(image=source_image_np, image0=target_image_np)
            source_image_np = augmentations["image0"]
            target_image_np = augmentations["image"]
        # create_folder_if_not_exists("TestConcatenatedImages")
        # source_save_path = f"TestConcatenatedImages/{source_image}"
        # save_image_as_jpg(source_image_np.permute(1, 2, 0).numpy(), source_save_path)
        # source_save_path2 = f"TestConcatenatedImages/{target_image}"
        # save_image_as_jpg(target_image_np.permute(1, 2, 0).numpy(), source_save_path2)
        return source_image_np, target_image_np
    

# Class to pass concatenated images
class ConcatenatedSourceTargetDataset(Dataset):
    def __init__(self, root_source:str, root_target:str, transform:Callable=None, resize_to: tuple=(256,256)):
        self.root_source = root_source
        self.root_target = root_target
        self.transform = transform
        self.resize_to = resize_to
        self.root_source_images = os.listdir(root_source)
        self.root_target_images = os.listdir(root_target)
        self.length_source = len(self.root_source_images)
        self.length_target = len(self.root_target_images)
        self.length_dataset = max(len(self.root_source_images)//2, len(self.root_target_images))
    def __len__(self):
        return self.length_dataset
    def __getitem__(self, index:int):
        source_image_1 = self.root_source_images[index % self.length_source]
        source_image_2 = self.root_source_images[(index + self.length_target) % self.length_source]
        # source_image_2 = self.root_source_images[(index + random.randint(1, self.length_source-1)) % self.length_source]
        target_image = self.root_target_images[index % self.length_target]
        # print(f"Indexes : {target_image} | {source_image_1} | {source_image_2}")
        source_path_1 = os.path.join(self.root_source, source_image_1)
        source_path_2 = os.path.join(self.root_source, source_image_2)
        target_path = os.path.join(self.root_target, target_image)
        source_img_1 = Image.open(source_path_1).convert("RGB")
        source_img_2 = Image.open(source_path_2).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(self.resize_to),
            transforms.ToTensor()
        ])
        source_image_tensor_1 = transform(source_img_1)
        source_image_tensor_2 = transform(source_img_2)
        source_image_tensor = torch.zeros_like(source_image_tensor_1)
        source_image_tensor[:, 0::2, ::2] = source_image_tensor_1[:, 0::2, ::2]  
        source_image_tensor[:, 0::2, 1::2] = source_image_tensor_2[:, 0::2, 1::2]  
        source_image_tensor[:, 1::2, ::2] = source_image_tensor_2[:, 1::2, ::2]  
        source_image_tensor[:, 1::2, 1::2] = source_image_tensor_1[:, 1::2, 1::2] 
        source_image_np = source_image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        target_image_np = np.array(Image.open(target_path).convert("RGB").resize(self.resize_to))
        if self.transform:
            augmentations = self.transform(image_source=source_image_np, image_target=target_image_np)
            source_image_np = augmentations["image_source"]
            target_image_np = augmentations["image_target"]

        create_folder_if_not_exists("TestConcatenatedImages")
        source_save_path = f"TestConcatenatedImages/{source_image_1.split('.')[0]}_{source_image_2.split('.')[0]}_combined.jpg"
        save_image_as_jpg(source_image_np.permute(1, 2, 0).numpy(), source_save_path)
        source_save_path2 = f"TestConcatenatedImages/{target_image}_combined.jpg"
        save_image_as_jpg(target_image_np.permute(1, 2, 0).numpy(), source_save_path2)
        # print("Concatenated Image saved")

        return source_image_np, target_image_np


# Function to combine images pixel-wise with alternating pattern
def combine_images(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    # print("combine_images")
    combined_source_image = torch.zeros_like(img1)
    combined_source_image[:, 0::2, ::2] = img1[:, 0::2, ::2]  
    combined_source_image[:, 0::2, 1::2] = img2[:, 0::2, 1::2]  
    combined_source_image[:, 1::2, ::2] = img2[:, 1::2, ::2]  
    combined_source_image[:, 1::2, 1::2] = img1[:, 1::2, 1::2] 
    return combined_source_image


# Function to reconstruct images from combined image
def reconstruct_images(combined_source_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    img1_reconstructed = torch.zeros_like(combined_source_image)
    img2_reconstructed = torch.zeros_like(combined_source_image)
    img1_reconstructed[:, 0::2, ::2] = combined_source_image[:, 0::2, ::2]
    img2_reconstructed[:, 0::2, 1::2] = combined_source_image[:, 0::2, 1::2]
    img2_reconstructed[:, 1::2, ::2] = combined_source_image[:, 1::2, ::2]
    img1_reconstructed[:, 1::2, 1::2] = combined_source_image[:, 1::2, 1::2]
    return img1_reconstructed, img2_reconstructed


# Function to save a numpy image as a JPEG file
def save_image_as_jpg(image_np: np.ndarray, filename: str):
    image = Image.fromarray(image_np.astype(np.uint8))
    image.save(filename, "JPEG")
    # print(f"Image saved as {filename}")


def create_folder_if_not_exists(folder_name: str) -> None:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# test_dataset = ConcatenatedSourceTargetDataset(root_source=config.TRAIN_DIR+"/"+config.SOURCE_DOMAIN, root_target=config.TRAIN_DIR+"/"+config.TARGET_DOMAIN, transform=config.transforms)
# print("Test dataset length : ", test_dataset.__len__())
# dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)