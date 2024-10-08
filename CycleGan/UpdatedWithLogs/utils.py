import random, torch, os, numpy as np
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