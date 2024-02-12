import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import models.cnn as cnn
from datasets.Dataloader import Caltech101
from torchvision.transforms import transforms
from tqdm import tqdm
from core.evaluate import AverageMeter
from core.trainer import Trainer
from utils.func import check_accuracy
import pandas as pd
import os

MODEL_DIR = '/Users/danielgomez/computer_vision_sg/clp1_cv_lp/clp_cnns/model'
ANNOTATIONS_DIR = '/Users/danielgomez/computer_vision_sg/clp1_cv_lp/clp_cnns/caltech-101/caltech101.csv'
IMAGES_DIR = '/Users/danielgomez/computer_vision_sg/clp1_cv_lp/clp_cnns/caltech-101'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
   
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# TODO: Define the hyperparameters you want to use for training
batch_size = 8
num_epochs = 10
learning_rate = 3e-4
num_classes = 102

# Load dataset
# TODO: Implement the transformations you consider suitable for this dataset
transform = transforms.ToTensor()
dataset = Caltech101(gt_path=ANNOTATIONS_DIR, imgs_source=IMAGES_DIR, transformations=transform)

# # TODO: Split the dataset into train, validation and test set
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [6401, 1372, 1372])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Initalize model
# # Here you should load your custom model for training
model = cnn.AlexNet(num_classes)

# loss and optimizer
# TODO: Choose the proper loss function and optimizer for training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = Trainer(
    model,
    num_epochs,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    model_dir=MODEL_DIR,
)

trainer.train()       


print(check_accuracy(train_loader, trainer.model, device)*100)
print(check_accuracy(val_loader, trainer.model, device)*100)

# if __name__ == "__main__": 
#     check_accuracy()
#     print('Hello world')