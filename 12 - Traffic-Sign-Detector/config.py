import os
import torch



BATCH_SIZE = 32  # increase / decrease according to GPU memeory
RESIZE_TO = 254 # resize the image for training and transforms
NUM_EPOCHS = 100 # number of epochs to train for
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True


# location to save model and plots
OUT_DIR = os.path.join("drive", "MyDrive", "Projects", "TrafficSign", "outputs")

# PATHS
DATA_PATH : str = "dataset"
IMGS_PATH : str = os.path.join(DATA_PATH, "images")
ANNT_PATH : str = os.path.join(DATA_PATH, "annotations")

# Early stopping patience
PATIENCE: int = 5

# MODEL OUTPUT
CLASSES = {
    1 : "speed limit 20 (prohibitory)",
    2 : "speed limit 30 (prohibitory)",
    3 : "speed limit 50 (prohibitory)",
    4 : "speed limit 60 (prohibitory)",
    5 : "speed limit 70 (prohibitory)",
    6 : "speed limit 80 (prohibitory)",
    7 : "restriction ends 80 (other)",
    8 : "speed limit 100 (prohibitory)",
    9 : "speed limit 120 (prohibitory)",
    10 : "no overtaking (prohibitory)",
    11 : "no overtaking (trucks) (prohibitory)",
    12 : "priority at next intersection (danger)",
    13 : "priority road (other)",
    14 : "give way (other)",
    15 : "stop (other)",
    16 : "no traffic both ways (prohibitory)",
    17 : "no trucks (prohibitory)",
    18 : "no entry (other)",
    19 : "danger (danger)",
    20 : "bend left (danger)",
    21 : "bend right (danger)",
    22 : "bend (danger)",
    23 : "uneven road (danger)",
    24 : "slippery road (danger)",
    25 : "road narrows (danger)",
    26 : "construction (danger)",
    27 : "traffic signal (danger)",
    28 : "pedestrian crossing (danger)",
    29 : "school crossing (danger)",
    30 : "cycles crossing (danger)",
    31 : "snow (danger)",
    32 : "animals (danger)",
    33 : "restriction ends (other)",
    34 : "go right (mandatory)",
    35 : "go left (mandatory)",
    36 : "go straight (mandatory)",
    37 : "go right or straight (mandatory)",
    38 : "go left or straight (mandatory)",
    39 : "keep right (mandatory)",
    40 : "keep left (mandatory)",
    41 : "roundabout (mandatory)",
    42 : "restriction ends (overtaking) (other)",
    43 : "restriction ends (overtaking (trucks)) (other)"
}

NUM_CLASSES: int = len(CLASSES) + 1 # +1 to account for background (0) by default in pytorch.