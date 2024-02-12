import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import models.cnn as cnn
from datasets.Dataloader import Caltech101
from torchvision.transforms import transforms
from tqdm import tqdm
from core.evaluate import AverageMeter
import pandas as pd
import os


MODEL_DIR = './model/checkpoint'
ANNOTATIONS_DIR = './caltech-101/caltech101.csv'
IMAGES_DIR = '/Users/danielgomez/computer_vision_sg/clp1_cv_lp/clp_cnns/caltech-101'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
   
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# TODO: Define the hyperparameters you want to use for training
batch_size = 8
num_epochs = 1
learning_rate = 3e-4
num_classes = 102



