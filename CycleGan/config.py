import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "resized_data/train"
VAL_DIR = "resized_data/validation"
BATCH_SIZE = 4
LEARNING_RATE = 0.003
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4

EXPERIMENT_NUMBER = 1
NUM_EPOCHS = 50
CHECKPOINT_EPOCH_NUMBER = 0
CHECKPOINT_SAVE_EPOCH_COUNT = 25
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_S = "gen_s.pth.tar"
CHECKPOINT_GEN_T = "gen_t.pth.tar"
CHECKPOINT_DISC_S = "disc_s.pth.tar"
CHECKPOINT_DISC_T = "disc_t.pth.tar"
SAVE_IMAGE_IDX = 50

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)