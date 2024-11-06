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
import utils

def create_folder_if_not_exists(folder_name: str) -> None:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def train_function_single(experiment_name:str, dataloader:DataLoader, epoch:int):
    loop = tqdm(dataloader, leave=True)
    print("inside loop")
    for idx, (source, target) in enumerate(loop):
        create_folder_if_not_exists(f"{experiment_name}_LoadedImages")
        print(f"images before loaded in epoch {epoch} index {idx}")
        source = source.to(config.DEVICE)
        target = target.to(config.DEVICE)
        print("Source Shape:", source.shape, "Target Shape:", target.shape)
        save_image(source*0.5+0.5, f"{experiment_name}_LoadedImages/{epoch}_{idx}_source.png")
        save_image(target*0.5+0.5, f"{experiment_name}_LoadedImages/{epoch}_{idx}_target.png")
        if config.ARNOLD_SCRAMBLE:
            scrambled_source = utils.arnold_transform_batch(source, config.SCRAMBLE_COUNT)
            source = scrambled_source
            print("Image scrambled")
        print(f"images before save")
        save_image(source*0.5+0.5, f"{experiment_name}_LoadedImages/{epoch}_{idx}_scrambled.png")
        if config.ARNOLD_SCRAMBLE:
            descrambled_source = utils.reverse_arnold_transform_batch(source, config.SCRAMBLE_COUNT)
            save_image(descrambled_source*0.5+0.5, f"{experiment_name}_LoadedImages/{epoch}_{idx}_descrambled_source.png")
        print(f"images saved in epoch {epoch} index {idx}")

def denormalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=image_tensor.device).view(1, 3, 1, 1)
    image_tensor = image_tensor * std + mean
    return image_tensor.clamp(0, 1)

def train_function_double(experiment_name:str, discriminator_T:Discriminator, discriminator_S:Discriminator, generator_S:Generator, generator_T:Generator, dataloader:DataLoader, discriminator_optimizer, generator_optimizer, L1_loss, MSE_loss, discriminator_scaler, generator_scaler, epoch:int, refine_technique:str = "4nn"):
    loop = tqdm(dataloader, leave=True)

    disc_t_loss_array = []
    disc_s_loss_array = []
    disc_total_loss_array = []
    gen_t_loss_array = []
    gen_s_loss_array = []
    cycle_t_loss_array = []
    cycle_s_loss_array = []
    gen_total_loss_array = []

    for idx, (source_1, source_2, target) in enumerate(loop):
        create_folder_if_not_exists(f"{experiment_name}_LoadedImages")

    # PREPARE IMAGES
        source_1 = source_1.to(config.DEVICE)
        source_2 = source_2.to(config.DEVICE)
        target = target.to(config.DEVICE)
        print(source_1.shape, source_2.shape, target.shape)
        source_1 = denormalize(source_1)
        source_2 = denormalize(source_2)
        target = denormalize(target)
        save_image(source_1, f"{experiment_name}_LoadedImages/{epoch}_{idx}_source_1.png")
        save_image(source_2, f"{experiment_name}_LoadedImages/{epoch}_{idx}_source_2.png")
        save_image(target, f"{experiment_name}_LoadedImages/{epoch}_{idx}_target.png")

    # FUSE 2 IMAGES WITH DEFINED TECHNIQUE - 4NN DONE, LSB TO BE IMPLEMENTED
        if (refine_technique == "4nn"):
            fused_image = utils.combine_images_pixel_wise_batch(source_1, source_2)
        elif ( refine_technique == "lsb"):
            # transform = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.ToTensor(),
            # ])
            # image_tensor_1 = (transform(source_1) * 255).to(torch.uint8)
            # image_tensor_2 = (transform(source_2) * 255).to(torch.uint8)
            fused_image = utils.combine_images_pixel_wise_msb_batch(source_1, source_2)
        else:
            print("undefined technique. process terminates")
            sys.exit()
        save_image(fused_image, f"{experiment_name}_LoadedImages/{epoch}_{idx}_fused_image.png")
        
    # SCRAMBLE INTERMEDIATE IMAGE
        if config.ARNOLD_SCRAMBLE:
            scrambled_intermediate = utils.arnold_transform_batch(fused_image, config.SCRAMBLE_COUNT)
            save_image(scrambled_intermediate, f"{experiment_name}_LoadedImages/{epoch}_{idx}_scrambled_image.png")
        else:
            scrambled_intermediate = fused_image
    

# =================================================================================== #
# =================================================================================== #
# =================================================================================== #

    # CYCLEGAN --> SCRAMBLED_INTERMEDIATE AS SOURCE | TARGET AS TARGET 
        with torch.cuda.amp.autocast():
            fake_target = generator_T(scrambled_intermediate)       # Generate a target image using source image
            discriminator_target_real = discriminator_T(target)
            discriminator_target_fake = discriminator_T(fake_target.detach())
            discriminator_target_real_loss = MSE_loss(discriminator_target_real, torch.ones_like(discriminator_target_real))
            discriminator_target_fake_loss = MSE_loss(discriminator_target_fake, torch.zeros_like(discriminator_target_fake))
            discriminator_target_loss = discriminator_target_real_loss + discriminator_target_fake_loss

            fake_source = generator_S(target)       # Generate a source image using target image
            discriminator_source_real = discriminator_S(scrambled_intermediate)
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
            cycle_source_loss = L1_loss(scrambled_intermediate, cycle_source)
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
            # if idx!=0:
            create_folder_if_not_exists(f"{experiment_name}_SavedImages")
            save_image(fake_target, f"{experiment_name}_SavedImages/target_{epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{idx}.png")
            save_image(fake_source, f"{experiment_name}_SavedImages/source_{epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{idx}.png")
            # save_image(fake_target*0.5+0.5, f"{experiment_name}_SavedImages/target_{epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{idx}.png")
            # save_image(fake_source*0.5+0.5, f"{experiment_name}_SavedImages/source_{epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{idx}.png")

    
# =================================================================================== #
# =================================================================================== #
# =================================================================================== #


    # DESCRAMBLE INTERMEDIATE IMAGE
        if config.ARNOLD_SCRAMBLE:        
            descrambled_intermediate = utils.reverse_arnold_transform_batch(cycle_source, config.SCRAMBLE_COUNT)
            save_image(descrambled_intermediate, f"{experiment_name}_LoadedImages/{epoch}_{idx}_descrambled_image.png")

    # DEFUSE 2 IMAGES WITH DEFINED TECHNIQUE - 4NN DONE, LSB TO BE IMPLEMENTED
        if (refine_technique == "4nn"):
            reconstruct_1, reconstruct_2 = utils.reconstruct_images_pixel_wise_batch(descrambled_intermediate)
            save_image(reconstruct_1, f"{experiment_name}_LoadedImages/{epoch}_{idx}_reconstruct_1.png")
            save_image(reconstruct_2, f"{experiment_name}_LoadedImages/{epoch}_{idx}_reconstruct_2.png")
            refine_1 = utils.replace_missing_pixels_with_neighbors_batch(reconstruct_1, 1)
            refine_2 = utils.replace_missing_pixels_with_neighbors_batch(reconstruct_2, 2)
            save_image(refine_1, f"{experiment_name}_LoadedImages/{epoch}_{idx}_refine_1.png")
            save_image(refine_2, f"{experiment_name}_LoadedImages/{epoch}_{idx}_refine_2.png")
        elif ( refine_technique == "lsb"):
            reconstruct_1, reconstruct_2 = utils.reconstruct_images_pixel_wise_msb_batch(descrambled_intermediate)
            print(reconstruct_1.shape, reconstruct_2.shape)
            save_image(reconstruct_1, f"{experiment_name}_LoadedImages/{epoch}_{idx}_reconstruct_msb_1.png")
            save_image(reconstruct_2, f"{experiment_name}_LoadedImages/{epoch}_{idx}_reconstruct_msb_2.png")
            refine_1 = reconstruct_1
            refine_2 = reconstruct_2

        print(f"images saved in epoch {epoch} index {idx}")


    disc_t_loss_value = sum(disc_t_loss_array) / len(disc_t_loss_array)
    disc_s_loss_value = sum(disc_s_loss_array) / len(disc_s_loss_array)
    disc_total_loss_value = sum(disc_total_loss_array) / len(disc_total_loss_array)
    gen_t_loss_value = sum(gen_t_loss_array) / len(gen_t_loss_array)
    gen_s_loss_value = sum(gen_s_loss_array) / len(gen_s_loss_array)
    cycle_t_loss_value = sum(cycle_t_loss_array) / len(cycle_t_loss_array)
    cycle_s_loss_value = sum(cycle_s_loss_array) / len(cycle_s_loss_array)
    gen_total_loss_value = sum(gen_total_loss_array) / len(gen_total_loss_array)

    # Calculate and Store Loss Values
    create_folder_if_not_exists(f"{experiment_name}_SavedLossFiles")
    
    disc_t_loss = f"{experiment_name}_SavedLossFiles/disc_t_loss.txt"
    disc_s_loss = f"{experiment_name}_SavedLossFiles/disc_s_loss.txt"
    disc_total_loss = f"{experiment_name}_SavedLossFiles/disc_total_loss.txt"
    with open(disc_t_loss, "a") as disc_t_loss_file, open(disc_s_loss, "a") as disc_s_loss_file, open(disc_total_loss, "a") as disc_total_loss_file:
        disc_t_loss_file.write(str(disc_t_loss_value) + "\n")
        disc_s_loss_file.write(str(disc_s_loss_value) + "\n")
        disc_total_loss_file.write(str(disc_total_loss_value) + "\n")

    gen_t_loss = f"{experiment_name}_SavedLossFiles/gen_t_loss.txt"
    gen_s_loss = f"{experiment_name}_SavedLossFiles/gen_s_loss.txt"
    with open(gen_t_loss, "a") as gen_t_loss_file, open(gen_s_loss, "a") as gen_s_loss_file:
        gen_t_loss_file.write(str(gen_t_loss_value) + "\n")
        gen_s_loss_file.write(str(gen_s_loss_value) + "\n")

    cycle_t_loss = f"{experiment_name}_SavedLossFiles/cycle_t_loss.txt"
    cycle_s_loss = f"{experiment_name}_SavedLossFiles/cycle_s_loss.txt"
    with open(cycle_t_loss, "a") as cycle_t_loss_file, open(cycle_s_loss, "a") as cycle_s_loss_file:
        cycle_t_loss_file.write(str(cycle_t_loss_value) + "\n")
        cycle_s_loss_file.write(str(cycle_s_loss_value) + "\n")
    
    gen_total_loss = f"{experiment_name}_SavedLossFiles/gen_total_loss.txt"
    with open(gen_total_loss, "a") as gen_total_loss_file:
        gen_total_loss_file.write(str(gen_total_loss_value) + "\n")
    


def train_experimental_model(experiment_name:str, root_source_path:str, root_source_2_path:str, root_target_path:str, transform_source_1:str, transform_source_2:str, transform_target:str, refine_technique:str) -> None:
    discriminator_S = Discriminator(in_channels=3).to(config.DEVICE)                # To classify images of source domain
    discriminator_T = Discriminator(in_channels=3).to(config.DEVICE)                # To classify images of target domain
    generator_S = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)    # To generate an image from source domain
    generator_T = Generator(image_channels=3, num_residuals=9).to(config.DEVICE)    # To generate an image from target domain
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
        load_checkpoint(f"{experiment_name}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_GEN_T}", generator_T, generator_optimizer, config.LEARNING_RATE)
        load_checkpoint(f"{experiment_name}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_GEN_S}", generator_S, generator_optimizer, config.LEARNING_RATE)
        load_checkpoint(f"{experiment_name}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_DISC_T}", discriminator_T, discriminator_optimizer, config.LEARNING_RATE)
        load_checkpoint(f"{experiment_name}_SavedModels/{config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{config.CHECKPOINT_DISC_S}", discriminator_S, discriminator_optimizer, config.LEARNING_RATE)
        print("Checkpoints loaded successfully")

    # For normal secret images -> two image
    dataset = SourceTargetDataset_TwoSecretImages(root_source=root_source_path, root_source2=root_source_2_path, root_target=root_target_path, transform_source1=transform_source_1, transform_source2=transform_source_2, transform_target=transform_target)
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"EPOCH - {epoch+1} / {config.NUM_EPOCHS}")
        train_function_double(
            experiment_name,
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
            epoch+1,
            refine_technique
        )
        if config.SAVE_MODEL and (epoch+1)%config.CHECKPOINT_SAVE_EPOCH_COUNT==0:
            create_folder_if_not_exists(f"{experiment_name}_SavedModels")
            save_checkpoint(experiment_name, generator_T, generator_optimizer, epoch+1, filename=config.CHECKPOINT_GEN_T)
            save_checkpoint(experiment_name, generator_S, generator_optimizer, epoch+1, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(experiment_name, discriminator_T, discriminator_optimizer, epoch+1, filename=config.CHECKPOINT_DISC_T)
            save_checkpoint(experiment_name, discriminator_S, discriminator_optimizer, epoch+1, filename=config.CHECKPOINT_DISC_S)
    print(f"TRAINING PROCESS SUCCESSFUL FOR {config.NUM_EPOCHS} EPOCHS in {experiment_name}")

def main():

    print("Experiment starts")
    # MAIN EXPERIMENT
    train_experimental_model(
        config.TEST_SOURCE2_TARGET_EXPERIMENT_NUMBER, 
        config.TEST_SOURCE2_TARGET_SOURCE_DOMAIN_TRAIN_DIR, 
        config.TEST_SOURCE2_TARGET_SOURCE2_DOMAIN_TRAIN_DIR, 
        config.TEST_SOURCE2_TARGET_TARGET_DOMAIN_TRAIN_DIR, 
        config.transform_source1, 
        config.transform_source2, 
        config.transform_target, 
        config.REFINE_TECHNIQUE
    )
    print("Experiment ends")

    # print("Experiment starts")
    # dataset = SourceTargetDataset(root_source=config.TEST_SOURCE2_TARGET_SOURCE_DOMAIN_TRAIN_DIR, root_target=config.TEST_SOURCE2_TARGET_TARGET_DOMAIN_TRAIN_DIR, transform=config.transforms)
    # dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    # for epoch in range(config.NUM_EPOCHS):
    #     print(f"EPOCH - {epoch+1} / {config.NUM_EPOCHS}")
    #     train_function_single(config.TEST_SOURCE2_TARGET_EXPERIMENT_NUMBER, dataloader, epoch+1)
    # print("Experiment ends")
    # return 0


    # print("Experiment starts")
    # dataset = SourceTargetDataset_TwoSecretImages(root_source=config.TEST_SOURCE2_TARGET_SOURCE_DOMAIN_TRAIN_DIR, root_source2=config.TEST_SOURCE2_TARGET_SOURCE2_DOMAIN_TRAIN_DIR, root_target=config.TEST_SOURCE2_TARGET_TARGET_DOMAIN_TRAIN_DIR, transform_source1=config.transform_source1, transform_source2=config.transform_source2, transform_target=config.transform_target)
    # dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    # for epoch in range(config.NUM_EPOCHS):
    #     print(f"EPOCH - {epoch+1} / {config.NUM_EPOCHS}")
    #     train_function_double(config.TEST_SOURCE2_TARGET_EXPERIMENT_NUMBER, dataloader, epoch+1)
    # print("Experiment ends")
    # return 0
        

if __name__ == "__main__":
    main()