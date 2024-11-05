import random, torch, os, numpy as np
from typing import Tuple
import torch.nn as nn
import config
import copy

def save_checkpoint(experiment_name:str, model, optimizer, epoch:int, filename:str):
    print(f"Saving checkpoint : {epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{experiment_name}_SavedModels/{epoch+config.CHECKPOINT_LOAD_EPOCH_NUMBER}_{filename}")


def load_checkpoint(checkpoint_file:str, model, optimizer, lr):
    print("Loading checkpoint ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###==========================###
###==== LSB MANIPULATION ====###
###==========================###

def lsb_Manipulate_fuse(source_image_np_1: np.ndarray, source_image_np_2: np.ndarray) -> np.ndarray:
    img1 = torch.tensor(source_image_np_1, dtype=torch.uint8).permute(2, 0, 1)  
    img2 = torch.tensor(source_image_np_2, dtype=torch.uint8).permute(2, 0, 1)  
    combined_image_tensor = combine_images_pixel_wise_msb(img1, img2)
    combined_image_np = combined_image_tensor.permute(1, 2, 0).numpy()
    return combined_image_np

def combine_images_pixel_wise_msb(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    img1 = img1.to(torch.uint8)
    img2 = img2.to(torch.uint8)
    combined_image_pixel_wise = torch.zeros_like(img1)
    img1_msb = img1 & 0xF0  # Most significant 4 bits of img1
    img2_msb = img2 & 0xF0  # Most significant 4 bits of img2
    combined_image_pixel_wise[:, ::2, ::2] = (img1_msb[:, ::2, ::2] | (img2_msb[:, ::2, ::2] >> 4))         # img1 MSBs in MSBs, img2 MSBs in LSBs
    combined_image_pixel_wise[:, ::2, 1::2] = (img2_msb[:, ::2, 1::2] | (img1_msb[:, ::2, 1::2] >> 4))      # img2 MSBs in MSBs, img1 MSBs in LSBs
    combined_image_pixel_wise[:, 1::2, ::2] = (img1_msb[:, 1::2, ::2] | (img2_msb[:, 1::2, ::2] >> 4))      # img1 MSBs in MSBs, img2 MSBs in LSBs
    combined_image_pixel_wise[:, 1::2, 1::2] = (img2_msb[:, 1::2, 1::2] | (img1_msb[:, 1::2, 1::2] >> 4))   # img2 MSBs in MSBs, img1 MSBs in LSBs
    return combined_image_pixel_wise

def reconstruct_images_pixel_wise_msb(combined_image_pixel_wise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    combined_image_pixel_wise = combined_image_pixel_wise.to(torch.uint8)
    img1_reconstructed_pixel_wise = torch.zeros_like(combined_image_pixel_wise)
    img2_reconstructed_pixel_wise = torch.zeros_like(combined_image_pixel_wise)
    img1_reconstructed_pixel_wise[:, ::2, ::2] = combined_image_pixel_wise[:, ::2, ::2] & 0xF0              # Get img1 MSBs
    img1_reconstructed_pixel_wise[:, ::2, 1::2] = (combined_image_pixel_wise[:, ::2, 1::2] & 0x0F) << 4     # Get img1 LSBs
    img2_reconstructed_pixel_wise[:, ::2, ::2] = (combined_image_pixel_wise[:, ::2, ::2] & 0x0F) << 4       # Get img2 MSBs
    img2_reconstructed_pixel_wise[:, ::2, 1::2] = combined_image_pixel_wise[:, ::2, 1::2] & 0xF0            # Get img2 MSBs
    img1_reconstructed_pixel_wise[:, 1::2, ::2] = combined_image_pixel_wise[:, 1::2, ::2] & 0xF0            # Get img1 MSBs
    img1_reconstructed_pixel_wise[:, 1::2, 1::2] = (combined_image_pixel_wise[:, 1::2, 1::2] & 0x0F) << 4   # Get img1 LSBs
    img2_reconstructed_pixel_wise[:, 1::2, ::2] = (combined_image_pixel_wise[:, 1::2, ::2] & 0x0F) << 4     # Get img2 MSBs
    img2_reconstructed_pixel_wise[:, 1::2, 1::2] = combined_image_pixel_wise[:, 1::2, 1::2] & 0xF0          # Get img2 MSBs
    return img1_reconstructed_pixel_wise, img2_reconstructed_pixel_wise


###==========================###
###===== 4NN REFINEMENT =====###
###==========================###

def concatenate_images_pixel_wise(source_image_np_1:np.ndarray, source_image_np_2:np.ndarray) -> np.ndarray:
    img1 = torch.tensor(source_image_np_1, dtype=torch.uint8).permute(2, 0, 1)  
    img2 = torch.tensor(source_image_np_2, dtype=torch.uint8).permute(2, 0, 1)  
    combined_image_tensor = combine_images_pixel_wise(img1, img2)
    combined_image_np = combined_image_tensor.permute(1, 2, 0).numpy()
    return combined_image_np

def combine_images_pixel_wise(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    combined_image_pixel_wise = torch.zeros_like(img1)
    combined_image_pixel_wise[:, 0::2, ::2] = img1[:, 0::2, ::2]    # img1 even rows, even cols
    combined_image_pixel_wise[:, 0::2, 1::2] = img2[:, 0::2, 1::2]  # img2 even rows, odd cols
    combined_image_pixel_wise[:, 1::2, ::2] = img2[:, 1::2, ::2]    # img2 odd rows, even cols
    combined_image_pixel_wise[:, 1::2, 1::2] = img1[:, 1::2, 1::2]  # img1 odd rows, odd cols
    return combined_image_pixel_wise

def reconstruct_images_pixel_wise(combined_image_pixel_wise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    img1_reconstructed_pixel_wise = torch.zeros_like(combined_image_pixel_wise)
    img2_reconstructed_pixel_wise = torch.zeros_like(combined_image_pixel_wise)
    img1_reconstructed_pixel_wise[:, 0::2, ::2] = combined_image_pixel_wise[:, 0::2, ::2]
    img2_reconstructed_pixel_wise[:, 0::2, 1::2] = combined_image_pixel_wise[:, 0::2, 1::2]
    img2_reconstructed_pixel_wise[:, 1::2, ::2] = combined_image_pixel_wise[:, 1::2, ::2]
    img1_reconstructed_pixel_wise[:, 1::2, 1::2] = combined_image_pixel_wise[:, 1::2, 1::2]
    img_1_refined_knn = replace_missing_pixels_with_neighbors(img1_reconstructed_pixel_wise, 1)
    img_2_refined_knn = replace_missing_pixels_with_neighbors(img2_reconstructed_pixel_wise, 2)
    return img_1_refined_knn, img_2_refined_knn

def replace_missing_pixels_with_neighbors(image_tensor: torch.Tensor, secret_image_number: int) -> torch.Tensor:
    channels, height, width = image_tensor.shape
    filled_image = image_tensor.clone()
    for i in range(height):
        for j in range(width):
            if secret_image_number == 1:
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    continue  
            elif secret_image_number == 2:
                if (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
                    continue  
            if torch.all(filled_image[:, i, j] == 0):
                neighbors = []
                if i - 1 >= 0 and torch.any(filled_image[:, i - 1, j] != 0):
                    neighbors.append(filled_image[:, i - 1, j])
                if i + 1 < height and torch.any(filled_image[:, i + 1, j] != 0):
                    neighbors.append(filled_image[:, i + 1, j])
                if j - 1 >= 0 and torch.any(filled_image[:, i, j - 1] != 0):
                    neighbors.append(filled_image[:, i, j - 1])
                if j + 1 < width and torch.any(filled_image[:, i, j + 1] != 0):
                    neighbors.append(filled_image[:, i, j + 1])
                if neighbors:
                    filled_image[:, i, j] = torch.mean(torch.stack(neighbors), dim=0)
    return filled_image


###=================================###
###===== ARNOLD TRANSFORMATION =====###
###=================================###

def arnold_transform_tensor(image_tensor: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    N = image_tensor.shape[1]  
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.zeros_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (x + y) % N
                y_new = (x + 2 * y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    return transformed_image

def reverse_arnold_transform_tensor(image_tensor: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    N = image_tensor.shape[1]  
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.zeros_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    return transformed_image

def arnold_transform(image_np: np.ndarray, iterations: int = 1) -> np.ndarray:
    image_tensor = torch.tensor(image_np, dtype=torch.uint8).permute(2, 0, 1)
    N = image_tensor.shape[1]
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.zeros_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (x + y) % N
                y_new = (x + 2 * y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    return transformed_image.permute(1, 2, 0).numpy()

def reverse_arnold_transform(image_np: np.ndarray, iterations: int = 1) -> np.ndarray:
    image_tensor = torch.tensor(image_np, dtype=torch.uint8).permute(2, 0, 1)
    N = image_tensor.shape[1]
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.zeros_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    return transformed_image.permute(1, 2, 0).numpy()

###=================================###
###===== MEAN PIXEL DIFFERENCE =====###
###=================================###

def calculate_mean_pixel_difference(image1: torch.Tensor, image2: torch.Tensor, absolute: bool = True) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Error: Tensors must have the same shape.")
    difference = image1 - image2
    if absolute:
        mean_difference = torch.mean(torch.abs(difference))
    else:
        mean_difference = torch.mean(difference ** 2)
    return mean_difference.item()  