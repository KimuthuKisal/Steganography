import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Adjust Dataset paths for Training data and Validation data - Only training data is used for now
TRAIN_DIR = "resized_data/train"
VAL_DIR = "resized_data/validation"

# Adjust followings as your wish - Optional
BATCH_SIZE = 4
LEARNING_RATE = 0.003
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4

# Define a new Experiment Number for new Experiment | Define the relevant Experiment number correctly when loading models
# Uses as a suffix for output folders and files
# Example : EXPERIMENT_NUMBER = 1  ->  Maintain a file to record data about the experiment
# Example : EXPERIMENT_NUMBER = "ImagesToGoogleMaps"
EXPERIMENT_NUMBER = 1

# Total number of epochs in the Training loop to be executed
# If new training   -> 1 to NUM_EPOCHS
# If load and train -> [CHECKPOINT_LOAD_EPOCH_NUMBER] to [CHECKPOINT_LOAD_EPOCH_NUMBER + NUM_EPOCHS]
NUM_EPOCHS = 50

# Define Model epoch number correctly to load models
CHECKPOINT_LOAD_EPOCH_NUMBER = 0

# Models are saved by this count continuously ( 25,50,75,100 )
# Uses as a suffix for output files
CHECKPOINT_SAVE_EPOCH_COUNT = 25

# If continueing previously trained model   -> True [Make sure to define CHECKPOINT_LOAD_EPOCH_NUMBER correctly]
# If new training                           -> False
LOAD_MODEL = False

# Saves model from given CHECKPOINT_SAVE_EPOCH_COUNT -> Usually True
SAVE_MODEL = True

# Constructed Images are saved from this count
# Uses as a suffix for output images
SAVE_IMAGE_IDX = 50

# Suffixes for generators and discriminators
CHECKPOINT_GEN_S = "gen_s.pth.tar"
CHECKPOINT_GEN_T = "gen_t.pth.tar"
CHECKPOINT_DISC_S = "disc_s.pth.tar"
CHECKPOINT_DISC_T = "disc_t.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)