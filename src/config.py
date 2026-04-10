import torch
import os

BASE_DIR = r"d:\Workspace\Learning\DL\food_detection"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
IMAGE_SIZE = 224
BATCH_SIZE = 16      
NUM_WORKERS = 2
TRAIN_SPLIT = 0.8

MODEL_NAME = "mobilenet_v2"
NUM_CLASSES = 2
PRETRAINED = True
FREEZE_BACKBONE = True   

EPOCHS = 15 
LEARNING_RATE = 0.003
WEIGHT_DECAY =  1e-4 
LOSS_FUNCTION = "cross_entropy"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "saved_models")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
