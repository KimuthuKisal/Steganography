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
EXPERIMENT_NUMBER = "ZZZZZZZZ_DELETE"
TEST_LOAD_EXPERIMENT_NUMBER = 100

# Adjust Dataset paths for Training data and Validation data - Only training data is used for now
SOURCE_DOMAIN_TRAIN_DIR = "horse_zebra/train/sourceDomain"
TARGET_DOMAIN_TRAIN_DIR = "horse_zebra/train/targetDomain"
SOURCE_DOMAIN_VALIDATION_DIR = "horse_zebra/validation/sourceDomain"
TARGET_DOMAIN_VALIDATION_DIR = "horse_zebra/validation/targetDomain"

# Adjust followings as your wish - Optional
BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 1
LEARNING_RATE = 0.0001
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
LAMBDA_RECONSTRUCTION = 10
NUM_WORKERS = 4
ARNOLD_SCRAMBLE = False
SCRAMBLE_COUNT = 8
REFINE_TECHNIQUE = '4nn'

# For Additional experiments after  main experiment -> Set flag True with an experiment name
# Use seperate train.py files to execute
# All other configurations may be same as the Original Experental values
# If it is essential to have different parameter values setup them inside the train.py files

EXPERIMENT_NUMBER_EXP02 = "ZZZZ_Paintarts"
EXPERIMENT_NUMBER_EXP02_FLAG = True
SOURCE_DOMAIN_TRAIN_DIR_EXP02 = "horse_zebra/train/sourceDomain"
TARGET_DOMAIN_TRAIN_DIR_EXP02 = "horse_zebra/train/targetDomain"

EXPERIMENT_NUMBER_EXP03 = "ZZZZ_Googlemaps"
EXPERIMENT_NUMBER_EXP03_FLAG = True
SOURCE_DOMAIN_TRAIN_DIR_EXP03 = "horse_zebra/train/sourceDomain"
TARGET_DOMAIN_TRAIN_DIR_EXP03 = "horse_zebra/train/targetDomain"

EXPERIMENT_NUMBER_EXP04 = "ZZZZ_Paintarts_Concatenated"
EXPERIMENT_NUMBER_EXP04_FLAG = True
SOURCE_DOMAIN_TRAIN_DIR_EXP04 = "horse_zebra/train/sourceDomain"
TARGET_DOMAIN_TRAIN_DIR_EXP04 = "horse_zebra/train/targetDomain"

EXPERIMENT_NUMBER_EXP05 = "ZZZZ_Googlemaps_Concatenated"
EXPERIMENT_NUMBER_EXP05_FLAG = True
SOURCE_DOMAIN_TRAIN_DIR_EXP05 = "horse_zebra/train/sourceDomain"
TARGET_DOMAIN_TRAIN_DIR_EXP05 = "horse_zebra/train/targetDomain"

# Total number of epochs in the Training loop to be executed
# If new training   -> 1 to NUM_EPOCHS
# If load and train -> [CHECKPOINT_LOAD_EPOCH_NUMBER] to [CHECKPOINT_LOAD_EPOCH_NUMBER + NUM_EPOCHS]
NUM_EPOCHS = 2

# Define Model epoch number correctly to load models
CHECKPOINT_LOAD_EPOCH_NUMBER = 0

# Models are saved by this count continuously ( if defined to 25 then 25,50,75,100 )
# Uses as a suffix for output files
CHECKPOINT_SAVE_EPOCH_COUNT = 1

# If continueing previously trained model   -> True [Make sure to define CHECKPOINT_LOAD_EPOCH_NUMBER correctly]
# If new training                           -> False
LOAD_MODEL = False

# Saves model from given CHECKPOINT_SAVE_EPOCH_COUNT -> Usually True
SAVE_MODEL = False

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

transforms_2secret_images = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image1", "image": "image1"},
)

transform_source1 = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

transform_source2 = A.Compose([
    A.Resize(width=256, height=256),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

transform_target = A.Compose([
    A.Resize(width=256, height=256),
    A.RandomRotate90(),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

TEST_SOURCE2_TARGET_EXPERIMENT_NUMBER = "TEST_SOURCE2_TARGET_EXPERIMENT"
TEST_SOURCE2_TARGET_EXPERIMENT_NUMBER_FLAG = True
TEST_SOURCE2_TARGET_SOURCE_DOMAIN_TRAIN_DIR = "test_source_2_target/source"
TEST_SOURCE2_TARGET_SOURCE2_DOMAIN_TRAIN_DIR = "test_source_2_target/source"
TEST_SOURCE2_TARGET_TARGET_DOMAIN_TRAIN_DIR = "test_source_2_target/target"