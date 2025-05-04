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

def combine_images_pixel_wise_msb_batch(img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> torch.Tensor:
    assert img1_batch.shape == img2_batch.shape, "Both image batches must have the same shape."
    batch_size, channels, height, width = img1_batch.shape
    source_1 = ((source_1 + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    source_2 = ((source_2 + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    combined_batch = torch.empty_like(img1_batch)
    for b in range(batch_size):
        # img1 = (img1_batch[b]*255).to(torch.uint8)
        # img2 = (img2_batch[b]*255).to(torch.uint8)
        img1 = img1_batch[b]
        img2 = img2_batch[b]
        img1_msb = img1 & 0xF0
        img2_msb = img2 & 0xF0
        combined_batch[b, :, ::2, ::2] = (img1_msb[:, ::2, ::2] | (img2_msb[:, ::2, ::2] >> 4))  # even row, even col
        combined_batch[b, :, ::2, 1::2] = (img2_msb[:, ::2, 1::2] | (img1_msb[:, ::2, 1::2] >> 4))  # even row, odd col
        combined_batch[b, :, 1::2, ::2] = (img2_msb[:, 1::2, ::2] | (img1_msb[:, 1::2, ::2] >> 4))  # odd row, even col
        combined_batch[b, :, 1::2, 1::2] = (img1_msb[:, 1::2, 1::2] | (img2_msb[:, 1::2, 1::2] >> 4))  # odd row, odd col
    return combined_batch

def reconstruct_images_pixel_wise_msb_batch(combined_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, channels, height, width = combined_batch.shape
    img1_reconstructed_batch = torch.empty_like(combined_batch)
    img2_reconstructed_batch = torch.empty_like(combined_batch)
    for b in range(batch_size):
        combined_image = combined_batch[b].to(torch.uint8)
        img1_reconstructed = torch.empty_like(combined_image)
        img2_reconstructed = torch.empty_like(combined_image)
        img1_reconstructed[:, ::2, ::2] = combined_image[:, ::2, ::2] & 0xF0             # even row, even col
        img2_reconstructed[:, ::2, ::2] = (combined_image[:, ::2, ::2] & 0x0F) << 4      # even row, even col
        img2_reconstructed[:, ::2, 1::2] = combined_image[:, ::2, 1::2] & 0xF0           # even row, odd col
        img1_reconstructed[:, ::2, 1::2] = (combined_image[:, ::2, 1::2] & 0x0F) << 4    # even row, odd col
        img2_reconstructed[:, 1::2, ::2] = combined_image[:, 1::2, ::2] & 0xF0           # odd row, even col
        img1_reconstructed[:, 1::2, ::2] = (combined_image[:, 1::2, ::2] & 0x0F) << 4    # odd row, even col
        img1_reconstructed[:, 1::2, 1::2] = combined_image[:, 1::2, 1::2] & 0xF0         # odd row, odd col
        img2_reconstructed[:, 1::2, 1::2] = (combined_image[:, 1::2, 1::2] & 0x0F) << 4  # odd row, odd col    
        img1_reconstructed_batch[b] = img1_reconstructed
        img2_reconstructed_batch[b] = img2_reconstructed 
    return img1_reconstructed_batch, img2_reconstructed_batch

def lsb_Manipulate_fuse(source_image_np_1: np.ndarray, source_image_np_2: np.ndarray) -> np.ndarray:
    img1 = torch.tensor(source_image_np_1, dtype=torch.uint8).permute(2, 0, 1)  
    img2 = torch.tensor(source_image_np_2, dtype=torch.uint8).permute(2, 0, 1)  
    combined_image_tensor = combine_images_pixel_wise_msb(img1, img2)
    combined_image_np = combined_image_tensor.permute(1, 2, 0).numpy()
    return combined_image_np

def combine_images_pixel_wise_msb(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    # Ensure input images are in uint8 format
    img1 = img1.to(torch.uint8)
    img2 = img2.to(torch.uint8)
    
    # Create a tensor for combined image with the same shape
    combined_image_pixel_wise = torch.empty_like(img1)
    
    # Extract Most Significant Bits (MSBs)
    img1_msb = img1 & 0xF0  # Most significant 4 bits of img1
    img2_msb = img2 & 0xF0  # Most significant 4 bits of img2
    
    # Combine the images pixel-wise
    combined_image_pixel_wise[:, ::2, ::2] = (img1_msb[:, ::2, ::2] | (img2_msb[:, ::2, ::2] >> 4))         # img1 MSBs in MSBs, img2 MSBs in LSBs
    combined_image_pixel_wise[:, ::2, 1::2] = (img2_msb[:, ::2, 1::2] | (img1_msb[:, ::2, 1::2] >> 4))      # img2 MSBs in MSBs, img1 MSBs in LSBs
    combined_image_pixel_wise[:, 1::2, ::2] = (img1_msb[:, 1::2, ::2] | (img2_msb[:, 1::2, ::2] >> 4))      # img1 MSBs in MSBs, img2 MSBs in LSBs
    combined_image_pixel_wise[:, 1::2, 1::2] = (img2_msb[:, 1::2, 1::2] | (img1_msb[:, 1::2, 1::2] >> 4))   # img2 MSBs in MSBs, img1 MSBs in LSBs
    
    return combined_image_pixel_wise

def reconstruct_images_pixel_wise_msb(combined_image_pixel_wise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure combined image is in uint8 format
    combined_image_pixel_wise = combined_image_pixel_wise.to(torch.uint8)
    
    # Create tensors for reconstructed images
    img1_reconstructed_pixel_wise = torch.empty_like(combined_image_pixel_wise)
    img2_reconstructed_pixel_wise = torch.empty_like(combined_image_pixel_wise)
    
    # Reconstruct the original images from the combined image
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
    del source_image_np_1, source_image_np_2
    return combined_image_np

# def combine_images_pixel_wise_batch(img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> torch.Tensor:
#     assert img1_batch.shape == img2_batch.shape, "Both image batches must have the same shape."
#     batch_size, channels, height, width = img1_batch.shape
#     combined_batch = torch.empty_like(img1_batch)
#     for b in range(batch_size):
#         combined_batch[b, :, 0::2, ::2] = img1_batch[b, :, 0::2, ::2]    # img1 even rows, even cols
#         combined_batch[b, :, 0::2, 1::2] = img2_batch[b, :, 0::2, 1::2]  # img2 even rows, odd cols
#         combined_batch[b, :, 1::2, ::2] = img2_batch[b, :, 1::2, ::2]    # img2 odd rows, even cols
#         combined_batch[b, :, 1::2, 1::2] = img1_batch[b, :, 1::2, 1::2]  # img1 odd rows, odd cols
#     del img1_batch, img2_batch
#     return combined_batch
def combine_images_pixel_wise_batch(img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> torch.Tensor:
    assert img1_batch.shape == img2_batch.shape, "Both image batches must have the same shape."
    combined = torch.empty_like(img1_batch)
    combined[:, :, 0::2, ::2] = img1_batch[:, :, 0::2, ::2]
    combined[:, :, 0::2, 1::2] = img2_batch[:, :, 0::2, 1::2]
    combined[:, :, 1::2, ::2] = img2_batch[:, :, 1::2, ::2]
    combined[:, :, 1::2, 1::2] = img1_batch[:, :, 1::2, 1::2]
    return combined

# def reconstruct_images_pixel_wise_batch(combined_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     batch_size, channels, height, width = combined_batch.shape
#     img1_reconstructed_batch = torch.empty_like(combined_batch)
#     img2_reconstructed_batch = torch.empty_like(combined_batch)
#     for b in range(batch_size):
#         img1_reconstructed_batch[b, :, 0::2, ::2] = combined_batch[b, :, 0::2, ::2]
#         img2_reconstructed_batch[b, :, 0::2, 1::2] = combined_batch[b, :, 0::2, 1::2]
#         img2_reconstructed_batch[b, :, 1::2, ::2] = combined_batch[b, :, 1::2, ::2]
#         img1_reconstructed_batch[b, :, 1::2, 1::2] = combined_batch[b, :, 1::2, 1::2]
#     del combined_batch
#     return img1_reconstructed_batch, img2_reconstructed_batch
def reconstruct_images_pixel_wise_batch(combined_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    img1 = torch.zeros_like(combined_batch)
    img2 = torch.zeros_like(combined_batch)
    img1[:, :, 0::2, ::2] = combined_batch[:, :, 0::2, ::2]
    img2[:, :, 0::2, 1::2] = combined_batch[:, :, 0::2, 1::2]
    img2[:, :, 1::2, ::2] = combined_batch[:, :, 1::2, ::2]
    img1[:, :, 1::2, 1::2] = combined_batch[:, :, 1::2, 1::2]
    return img1, img2

def combine_images_pixel_wise(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    combined_image_pixel_wise = torch.empty_like(img1)
    combined_image_pixel_wise[:, 0::2, ::2] = img1[:, 0::2, ::2]    # img1 even rows, even cols
    combined_image_pixel_wise[:, 0::2, 1::2] = img2[:, 0::2, 1::2]  # img2 even rows, odd cols
    combined_image_pixel_wise[:, 1::2, ::2] = img2[:, 1::2, ::2]    # img2 odd rows, even cols
    combined_image_pixel_wise[:, 1::2, 1::2] = img1[:, 1::2, 1::2]  # img1 odd rows, odd cols
    del img1, img2
    return combined_image_pixel_wise

def reconstruct_images_pixel_wise(combined_image_pixel_wise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    img1_reconstructed_pixel_wise = torch.empty_like(combined_image_pixel_wise)
    img2_reconstructed_pixel_wise = torch.empty_like(combined_image_pixel_wise)
    img1_reconstructed_pixel_wise[:, 0::2, ::2] = combined_image_pixel_wise[:, 0::2, ::2]
    img2_reconstructed_pixel_wise[:, 0::2, 1::2] = combined_image_pixel_wise[:, 0::2, 1::2]
    img2_reconstructed_pixel_wise[:, 1::2, ::2] = combined_image_pixel_wise[:, 1::2, ::2]
    img1_reconstructed_pixel_wise[:, 1::2, 1::2] = combined_image_pixel_wise[:, 1::2, 1::2]
    del combined_image_pixel_wise
    return img1_reconstructed_pixel_wise, img2_reconstructed_pixel_wise
    # img_1_refined_knn = replace_missing_pixels_with_neighbors(img1_reconstructed_pixel_wise, 1)
    # img_2_refined_knn = replace_missing_pixels_with_neighbors(img2_reconstructed_pixel_wise, 2)
    # return img_1_refined_knn, img_2_refined_knn

def replace_missing_pixels_with_neighbors_batch(image_batch: torch.Tensor, secret_image_number: int) -> torch.Tensor:
    batch_size, channels, height, width = image_batch.shape
    filled = image_batch.clone()
    mask = torch.zeros((height, width), dtype=torch.bool, device=image_batch.device)
    if secret_image_number == 1:
        mask[0::2, ::2] = True
        mask[1::2, 1::2] = True
    else:
        mask[0::2, 1::2] = True
        mask[1::2, ::2] = True
    missing_mask = (filled == 0).all(1) & mask.unsqueeze(0)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = torch.roll(filled, shifts=(dx, dy), dims=(2, 3))
        valid_shift = (shifted != 0).any(1)
        filled[missing_mask & valid_shift] += shifted[missing_mask & valid_shift]

    counts = (filled != 0).float().sum(1, keepdim=True)
    counts[counts == 0] = 1
    return filled / counts

# def replace_missing_pixels_with_neighbors_batch(image_batch: torch.Tensor, secret_image_number: int) -> torch.Tensor:
#     batch_size, channels, height, width = image_batch.shape
#     filled_batch = image_batch.clone()
#     for b in range(batch_size):
#         for i in range(height):
#             for j in range(width):
#                 if secret_image_number == 1:
#                     if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
#                         continue  
#                 elif secret_image_number == 2:
#                     if (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
#                         continue  
#                 if torch.all(filled_batch[b, :, i, j] == 0):  
#                     neighbors = []
#                     if i - 1 >= 0 and torch.any(filled_batch[b, :, i - 1, j] != 0):
#                         neighbors.append(filled_batch[b, :, i - 1, j])
#                     if i + 1 < height and torch.any(filled_batch[b, :, i + 1, j] != 0):
#                         neighbors.append(filled_batch[b, :, i + 1, j])
#                     if j - 1 >= 0 and torch.any(filled_batch[b, :, i, j - 1] != 0):
#                         neighbors.append(filled_batch[b, :, i, j - 1])
#                     if j + 1 < width and torch.any(filled_batch[b, :, i, j + 1] != 0):
#                         neighbors.append(filled_batch[b, :, i, j + 1])
#                     if neighbors:
#                         filled_batch[b, :, i, j] = (torch.mean(torch.stack(neighbors), dim=0))
#     del image_batch
#     return filled_batch



###=================================###
###===== ARNOLD TRANSFORMATION =====###
###=================================###

def arnold_transform_tensor(image_tensor: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    N = image_tensor.shape[1]  
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.empty_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (x + y) % N
                y_new = (x + 2 * y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    del image_tensor
    return transformed_image

def reverse_arnold_transform_tensor(image_tensor: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    N = image_tensor.shape[1]  
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.empty_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    del image_tensor
    return transformed_image

def arnold_transform(image_np: np.ndarray, iterations: int = 1) -> np.ndarray:
    image_tensor = torch.tensor(image_np, dtype=torch.uint8).permute(2, 0, 1)
    N = image_tensor.shape[1]
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.empty_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (x + y) % N
                y_new = (x + 2 * y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    del image_np
    return transformed_image.permute(1, 2, 0).numpy()

def reverse_arnold_transform(image_np: np.ndarray, iterations: int = 1) -> np.ndarray:
    image_tensor = torch.tensor(image_np, dtype=torch.uint8).permute(2, 0, 1)
    N = image_tensor.shape[1]
    transformed_image = image_tensor.clone()
    for _ in range(iterations):
        new_image = torch.empty_like(transformed_image)
        for x in range(N):
            for y in range(N):
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                new_image[:, x_new, y_new] = transformed_image[:, x, y]
        transformed_image = new_image
    del image_np
    return transformed_image.permute(1, 2, 0).numpy()

# def arnold_transform_batch(image_batch: torch.Tensor, iterations: int = 1) -> torch.Tensor:
#     B, C, H, W = image_batch.shape
#     assert H == W, "Arnold transform requires square images."
#     N = H
#     transformed_batch = image_batch.clone()

#     for _ in range(iterations):
#         new_batch = torch.empty_like(transformed_batch)
#         for x in range(N):
#             for y in range(N):
#                 x_new = (x + y) % N
#                 y_new = (x + 2 * y) % N
#                 new_batch[:, :, x_new, y_new] = transformed_batch[:, :, x, y]
#         transformed_batch = new_batch

#     del image_batch
#     return transformed_batch

# def reverse_arnold_transform_batch(image_batch: torch.Tensor, iterations: int = 1) -> torch.Tensor:
#     B, C, H, W = image_batch.shape
#     assert H == W, "Arnold transform requires square images."
#     N = H
#     transformed_batch = image_batch.clone()

#     for _ in range(iterations):
#         new_batch = torch.empty_like(transformed_batch)
#         for x in range(N):
#             for y in range(N):
#                 x_new = (2 * x - y) % N
#                 y_new = (-x + y) % N
#                 new_batch[:, :, x_new, y_new] = transformed_batch[:, :, x, y]
#         transformed_batch = new_batch

#     del image_batch
#     return transformed_batch

def arnold_transform_tensor_batch(images: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    b, c, h, w = images.shape
    out = images.clone()
    for _ in range(iterations):
        x = torch.arange(h).view(-1, 1).expand(h, w)
        y = torch.arange(w).view(1, -1).expand(h, w)
        x_new = (x + y) % h
        y_new = (x + 2 * y) % w
        out = out[:, :, x_new, y_new]
    return out

def reverse_arnold_transform_tensor_batch(images: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    b, c, h, w = images.shape
    out = images.clone()
    for _ in range(iterations):
        x = torch.arange(h).view(-1, 1).expand(h, w)
        y = torch.arange(w).view(1, -1).expand(h, w)
        x_new = (2 * x - y) % h
        y_new = (-x + y) % w
        out = out[:, :, x_new, y_new]
    return out


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