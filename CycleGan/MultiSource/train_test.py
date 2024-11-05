import torch
from dataset import SourceTargetDataset, SourceTargetDataset_TwoSecretImages
import sys
from utils import save_checkpoint, load_checkpoint
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader  
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import os


def create_folder_if_not_exists(folder_name: str) -> None:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def train_function_single(experiment_name:str, dataloader:DataLoader, epoch:int):
    loop = tqdm(dataloader, leave=True)
    print("inside loop")
    for idx, (source, target) in enumerate(loop):
        print(f"images before loaded in epoch {epoch} index {idx}")
        source = source.to(config.DEVICE)
        target = target.to(config.DEVICE)
        print(source.shape, target.shape)
        create_folder_if_not_exists(f"{experiment_name}_LoadedImages")
        print(f"images before save")
        save_image(source*0.5+0.5, f"{experiment_name}_LoadedImages/source_{epoch}_{idx}.png")
        save_image(target*0.5+0.5, f"{experiment_name}_LoadedImages/target_{epoch}_{idx}.png")
        print(f"images saved in epoch {epoch} index {idx}")

def denormalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=image_tensor.device).view(1, 3, 1, 1)
    image_tensor = image_tensor * std + mean
    return image_tensor.clamp(0, 1)

def train_function_double(experiment_name:str, dataloader:DataLoader, epoch:int):
    loop = tqdm(dataloader, leave=True)
    print("inside loop")
    for idx, (source_1, source_2, target) in enumerate(loop):
        print(f"images before loaded in epoch {epoch} index {idx}")
        source_1 = source_1.to(config.DEVICE)
        source_2 = source_2.to(config.DEVICE)
        target = target.to(config.DEVICE)
        print(source_1.shape, source_2.shape, target.shape)
        source_1 = denormalize(source_1)
        source_2 = denormalize(source_2)
        target = denormalize(target)
        create_folder_if_not_exists(f"{experiment_name}_LoadedImages")
        print(f"images before save")
        save_image(source_1, f"{experiment_name}_LoadedImages/source_1_{epoch}_{idx}.png")
        save_image(source_2, f"{experiment_name}_LoadedImages/source_2_{epoch}_{idx}.png")
        save_image(target, f"{experiment_name}_LoadedImages/target_{epoch}_{idx}.png")
        print(f"images saved in epoch {epoch} index {idx}")


def main():

    # print("Experiment starts")
    # dataset = SourceTargetDataset(root_source=config.TEST_SOURCE2_TARGET_SOURCE_DOMAIN_TRAIN_DIR, root_target=config.TEST_SOURCE2_TARGET_TARGET_DOMAIN_TRAIN_DIR, transform=config.transforms)
    # dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    # for epoch in range(config.NUM_EPOCHS):
    #     print(f"EPOCH - {epoch+1} / {config.NUM_EPOCHS}")
    #     train_function_single(config.TEST_SOURCE2_TARGET_EXPERIMENT_NUMBER, dataloader, epoch+1)
    # print("Experiment ends")
    # return 0


    print("Experiment starts")
    dataset = SourceTargetDataset_TwoSecretImages(root_source=config.TEST_SOURCE2_TARGET_SOURCE_DOMAIN_TRAIN_DIR, root_source2=config.TEST_SOURCE2_TARGET_SOURCE2_DOMAIN_TRAIN_DIR,root_target=config.TEST_SOURCE2_TARGET_TARGET_DOMAIN_TRAIN_DIR, transform_source1=config.transform_source1, transform_source2=config.transform_source2, transform_target=config.transform_target)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    for epoch in range(config.NUM_EPOCHS):
        print(f"EPOCH - {epoch+1} / {config.NUM_EPOCHS}")
        train_function_double(config.TEST_SOURCE2_TARGET_EXPERIMENT_NUMBER, dataloader, epoch+1)
    print("Experiment ends")
    return 0
        

if __name__ == "__main__":
    main()