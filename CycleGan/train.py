import torch
from dataset import SourceTargetDataset
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


def train_function(discriminator_T:Discriminator, discriminator_S:Discriminator, generator_S:Generator, generator_T:Generator, dataloader:DataLoader, 
                   discriminator_optimizer, generator_optimizer, L1_loss, MSE_loss, discriminator_scaler, generator_scaler, epoch:int):
    loop = tqdm(dataloader, leave=True)

    disc_t_loss_array = []
    disc_s_loss_array = []
    disc_total_loss_array = []
    gen_t_loss_array = []
    gen_s_loss_array = []
    cycle_t_loss_array = []
    cycle_s_loss_array = []
    gen_total_loss_array = []

    for idx, (source, target) in enumerate(loop):
        source = source.to(config.DEVICE)
        target = target.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_target = generator_T(source)
            discriminator_target_real = discriminator_T(source)
            discriminator_target_fake = discriminator_T(fake_target.detach())
            discriminator_target_real_loss = MSE_loss(discriminator_target_real, torch.ones_like(discriminator_target_real))
            discriminator_target_fake_loss = MSE_loss(discriminator_target_fake, torch.zeros_like(discriminator_target_fake))
            discriminator_target_loss = discriminator_target_real_loss + discriminator_target_fake_loss

            fake_source = generator_S(target)
            discriminator_source_real = discriminator_S(target)
            discriminator_source_fake = discriminator_S(fake_source.detach())
            discriminator_source_real_loss = MSE_loss(discriminator_source_real, torch.ones_like(discriminator_source_real))
            discriminator_source_fake_loss = MSE_loss(discriminator_source_fake, torch.zeros_like(discriminator_source_fake))
            discriminator_source_loss = discriminator_source_real_loss + discriminator_source_fake_loss

            discriminator_loss = (discriminator_target_loss + discriminator_source_loss)/2
        
            disc_t_loss_array.append(discriminator_target_loss)
            disc_s_loss_array.append(discriminator_source_loss)
            disc_total_loss_array.append(discriminator_loss)

        discriminator_optimizer.zero_grad()
        discriminator_scaler.scale(discriminator_loss).backward()
        discriminator_scaler.step(discriminator_optimizer)
        discriminator_scaler.update()

        # Train Generators 
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            discriminator_target_fake = discriminator_T(fake_target)
            discriminator_source_fake = discriminator_S(fake_source)
            generator_target_loss = MSE_loss(discriminator_target_fake, torch.ones_like(discriminator_target_fake))
            generator_source_loss = MSE_loss(discriminator_source_fake, torch.ones_like(discriminator_source_fake))

            # cycle loss
            cycle_source = generator_S(fake_target)
            cycle_target = generator_T(fake_source)
            cycle_source_loss = L1_loss(source, cycle_source)
            cycle_target_loss = L1_loss(target, cycle_target)

            # identity loss - commented as LAMBDA_IDENTITY = 0.0
            # identity_source = generator_S(source)
            # identity_target = generator_T(target)
            # identity_source_loss = L1_loss(source, identity_source)
            # identity_target_loss = L1_loss(target, identity_target)

            generator_loss = (
                generator_source_loss + generator_target_loss  
                + cycle_source_loss*config.LAMBDA_CYCLE + cycle_target_loss*config.LAMBDA_CYCLE 
                # + identity_source_loss*config.LAMBDA_IDENTITY + identity_target_loss*config.LAMBDA_IDENTITY
            )

            gen_t_loss_array.append(generator_target_loss)
            gen_s_loss_array.append(generator_source_loss)
            cycle_t_loss_array.append(cycle_target_loss)
            cycle_s_loss_array.append(cycle_source_loss)
            gen_total_loss_array.append(generator_loss)
        
        generator_optimizer.zero_grad()
        generator_scaler.scale(generator_loss).backward()
        generator_scaler.step(generator_optimizer)
        generator_scaler.update()

        if idx%config.SAVE_IMAGE_IDX == 0:
            if idx!=0:
                create_folder_if_not_exists(f"{config.EXPERIMENT_NUMBER}_SavedImages")
                save_image(fake_target*0.5+0.5, f"{config.EXPERIMENT_NUMBER}_SavedImages/target_{epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{idx}.png")
                save_image(fake_source*0.5+0.5, f"{config.EXPERIMENT_NUMBER}_SavedImages/source_{epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{idx}.png")

    disc_t_loss_value = sum(disc_t_loss_array) / len(disc_t_loss_array)
    disc_s_loss_value = sum(disc_s_loss_array) / len(disc_s_loss_array)
    disc_total_loss_value = sum(disc_total_loss_array) / len(disc_total_loss_array)
    gen_t_loss_value = sum(gen_t_loss_array) / len(gen_t_loss_array)
    gen_s_loss_value = sum(gen_s_loss_array) / len(gen_s_loss_array)
    cycle_t_loss_value = sum(cycle_t_loss_array) / len(cycle_t_loss_array)
    cycle_s_loss_value = sum(cycle_s_loss_array) / len(cycle_s_loss_array)
    gen_total_loss_value = sum(gen_total_loss_array) / len(gen_total_loss_array)

    # Calculate and Store Loss Values
    create_folder_if_not_exists(f"{config.EXPERIMENT_NUMBER}_SavedLossFiles")
    disc_t_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/disc_t_loss.txt"
    disc_s_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/disc_s_loss.txt"
    disc_total_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/disc_total_loss.txt"
    with open(disc_t_loss, "a") as disc_t_loss_file, open(disc_s_loss, "a") as disc_s_loss_file, open(disc_total_loss, "a") as disc_total_loss_file:
        disc_t_loss_file.write(str(disc_t_loss_value) + "\n")
        disc_s_loss_file.write(str(disc_s_loss_value) + "\n")
        disc_total_loss_file.write(str(disc_total_loss_value) + "\n")

    gen_t_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/gen_t_loss.txt"
    gen_s_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/gen_s_loss.txt"
    with open(gen_t_loss, "a") as gen_t_loss_file, open(gen_s_loss, "a") as gen_s_loss_file:
        gen_t_loss_file.write(str(gen_t_loss_value) + "\n")
        gen_s_loss_file.write(str(gen_s_loss_value) + "\n")

    cycle_t_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/cycle_t_loss.txt"
    cycle_s_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/cycle_s_loss.txt"
    with open(cycle_t_loss, "a") as cycle_t_loss_file, open(cycle_s_loss, "a") as cycle_s_loss_file:
        cycle_t_loss_file.write(str(cycle_t_loss_value) + "\n")
        cycle_s_loss_file.write(str(cycle_s_loss_value) + "\n")
    
    gen_total_loss = f"{config.EXPERIMENT_NUMBER}_SavedLossFiles/gen_total_loss.txt"
    with open(gen_total_loss, "a") as gen_total_loss_file:
        gen_total_loss_file.write(str(gen_total_loss_value) + "\n")


def main():
    discriminator_S = Discriminator(in_channels=3).to(config.DEVICE)
    discriminator_T = Discriminator(in_channels=3).to(config.DEVICE)
    generator_S = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)
    generator_T = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)
    discriminator_optimizer = optim.Adam(
        list(discriminator_T.parameters()) + list(discriminator_S.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999)
    )
    generator_optimizer = optim.Adam(
        list(generator_S.parameters()) + list(generator_T.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999)
    )

    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    if config.LOAD_MODEL:
        if config.CHECKPOINT_LOAD_EPOCH_NUMBER==0:
            print("Provide the epoch number to load models")
            sys.exit()            
        load_checkpoint(f"{config.EXPERIMENT_NUMBER}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_GEN_T}", generator_T, generator_optimizer, config.LEARNING_RATE)
        load_checkpoint(f"{config.EXPERIMENT_NUMBER}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_GEN_S}", generator_S, generator_optimizer, config.LEARNING_RATE)
        load_checkpoint(f"{config.EXPERIMENT_NUMBER}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_DISC_T}", discriminator_T, discriminator_optimizer, config.LEARNING_RATE)
        load_checkpoint(f"{config.EXPERIMENT_NUMBER}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_DISC_S}", discriminator_S, discriminator_optimizer, config.LEARNING_RATE)
        print("Checkpoints loaded successfully")

    ## For normal secret images -> single image
    # dataset = SourceTargetDataset(root_source=config.TRAIN_DIR+"/"+config.SOURCE_DOMAIN, root_target=config.TRAIN_DIR+"/"+config.TARGET_DOMAIN, transform=config.transforms)
    
    # For fused secret images -> concatenated two images
    dataset = ConcatenatedSourceTargetDataset(root_source=config.TRAIN_DIR+"/"+config.SOURCE_DOMAIN, root_target=config.TRAIN_DIR+"/"+config.TARGET_DOMAIN, transform=config.transforms)
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"EPOCH - {epoch+1} / {config.NUM_EPOCHS}")
        train_function(
            discriminator_T, 
            discriminator_S, 
            generator_S, 
            generator_T, 
            dataloader, 
            discriminator_optimizer, 
            generator_optimizer, 
            L1_loss, 
            MSE_loss, 
            discriminator_scaler, 
            generator_scaler,
            epoch+1
        )
        if config.SAVE_MODEL and (epoch+1)%config.CHECKPOINT_SAVE_EPOCH_COUNT==0:
            create_folder_if_not_exists(f"{config.EXPERIMENT_NUMBER}_SavedModels")
            save_checkpoint(generator_T, generator_optimizer, epoch+1, filename=config.CHECKPOINT_GEN_T)
            save_checkpoint(generator_S, generator_optimizer, epoch+1, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(discriminator_T, discriminator_optimizer, epoch+1, filename=config.CHECKPOINT_DISC_T)
            save_checkpoint(discriminator_S, discriminator_optimizer, epoch+1, filename=config.CHECKPOINT_DISC_S)
    print(f"TRAINING PROCESS SUCCESSFUL FOR {config.NUM_EPOCHS} EPOCHS")

if __name__ == "__main__":
    main()