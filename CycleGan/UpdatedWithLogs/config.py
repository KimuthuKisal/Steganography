import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRAIN_DIR = "horse_zebra/train"
# VAL_DIR = "horse_zebra/validation"
# SOURCE_DOMAIN = "sourceDomain"
# TARGET_DOMAIN = "targetDomain"

# Define a new Experiment Number for new Experiment | Define the relevant Experiment number correctly when loading models
# Uses as a suffix for output folders and files
# Example : EXPERIMENT_NUMBER = 1  ->  Maintain a file to record data about the experiment
# Example : EXPERIMENT_NUMBER = "ImagesToGoogleMaps"
EXPERIMENT_NUMBER = "UPDATED_EPOCH_1000_horse_zebra"
TEST_LOAD_EXPERIMENT_NUMBER = 100

# Adjust Dataset paths for Training data and Validation data - Only training data is used for now
SOURCE_DOMAIN_TRAIN_DIR = "horse_zebra/train/sourceDomain"
TARGET_DOMAIN_TRAIN_DIR = "horse_zebra/train/targetDomain"
SOURCE_DOMAIN_VALIDATION_DIR = "horse_zebra/validation/sourceDomain"
TARGET_DOMAIN_VALIDATION_DIR = "horse_zebra/validation/targetDomain"

# Adjust followings as your wish - Optional
BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 1
LEARNING_RATE = 0.0002
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4

# For Additional experiments after  main experiment -> Set flag True with an experiment name
# Use seperate train.py files to execute
# All other configurations may be same as the Original Experental values
# If it is essential to have different parameter values setup them inside the train.py files

EXPERIMENT_NUMBER_EXP02 = "UPDATED_EPOCH_1000_Paintarts"
EXPERIMENT_NUMBER_EXP02_FLAG = False
SOURCE_DOMAIN_TRAIN_DIR_EXP02 = "preprocessed_images/clean_secret_images_500"
TARGET_DOMAIN_TRAIN_DIR_EXP02 = "preprocessed_images/paintarts_500"

EXPERIMENT_NUMBER_EXP03 = "UPDATED_EPOCH_1000_Googlemaps"
EXPERIMENT_NUMBER_EXP03_FLAG = False
SOURCE_DOMAIN_TRAIN_DIR_EXP03 = "preprocessed_images/clean_secret_images_500"
TARGET_DOMAIN_TRAIN_DIR_EXP03 = "preprocessed_images/googlemaps_500"

EXPERIMENT_NUMBER_EXP04 = "UPDATED_EPOCH_1000_Paintarts_Concatenated"
EXPERIMENT_NUMBER_EXP04_FLAG = False
SOURCE_DOMAIN_TRAIN_DIR_EXP04 = "preprocessed_images/concatenated_images_500"
TARGET_DOMAIN_TRAIN_DIR_EXP04 = "preprocessed_images/paintarts_500"

EXPERIMENT_NUMBER_EXP05 = "UPDATED_EPOCH_1000_Googlemaps_Concatenated"
EXPERIMENT_NUMBER_EXP05_FLAG = False
SOURCE_DOMAIN_TRAIN_DIR_EXP05 = "preprocessed_images/concatenated_images_500"
TARGET_DOMAIN_TRAIN_DIR_EXP05 = "preprocessed_images/googlemaps_500"

EXPERIMENT_NUMBER_EXP06 = "UPDATED_EPOCH_1000_Paintarts_Concatenated_3500"
EXPERIMENT_NUMBER_EXP06_FLAG = False
SOURCE_DOMAIN_TRAIN_DIR_EXP06 = "preprocessed_images/concatenated_images_3500"
TARGET_DOMAIN_TRAIN_DIR_EXP06 = "preprocessed_images/paintarts_500"

EXPERIMENT_NUMBER_EXP07 = "UPDATED_EPOCH_1000_Googlemaps_Concatenated_1000"
EXPERIMENT_NUMBER_EXP07_FLAG = False
SOURCE_DOMAIN_TRAIN_DIR_EXP07 = "preprocessed_images/concatenated_images_1000"
TARGET_DOMAIN_TRAIN_DIR_EXP07 = "preprocessed_images/googlemaps_500"

EXPERIMENT_NUMBER_EXP08 = "UPDATED_EPOCH_1000_AppleOrange_3500"
EXPERIMENT_NUMBER_EXP08_FLAG = True
SOURCE_DOMAIN_TRAIN_DIR_EXP08 = "preprocessed_images/AppleOrange"
TARGET_DOMAIN_TRAIN_DIR_EXP08 = "preprocessed_images/googlemaps_500"

EXPERIMENT_NUMBER_EXP09 = "UPDATED_EPOCH_1000_AppleOrange_Concatenated_1000"
EXPERIMENT_NUMBER_EXP09_FLAG = True
SOURCE_DOMAIN_TRAIN_DIR_EXP09 = "preprocessed_images/AppleOrangeConcatenated"
TARGET_DOMAIN_TRAIN_DIR_EXP09 = "preprocessed_images/googlemaps_500"

# Total number of epochs in the Training loop to be executed
# If new training   -> 1 to NUM_EPOCHS
# If load and train -> [CHECKPOINT_LOAD_EPOCH_NUMBER] to [CHECKPOINT_LOAD_EPOCH_NUMBER + NUM_EPOCHS]
NUM_EPOCHS = 1000

# Define Model epoch number correctly to load models
CHECKPOINT_LOAD_EPOCH_NUMBER = 0

# Models are saved by this count continuously ( if defined to 25 then 25,50,75,100 )
# Uses as a suffix for output files
CHECKPOINT_SAVE_EPOCH_COUNT = 25

# If continueing previously trained model   -> True [Make sure to define CHECKPOINT_LOAD_EPOCH_NUMBER correctly]
# If new training                           -> False
LOAD_MODEL = False

# Saves model from given CHECKPOINT_SAVE_EPOCH_COUNT -> Usually True
SAVE_MODEL = True

# Constructed Images are saved from this count
# Uses as a suffix for output images
SAVE_IMAGE_IDX = 25

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