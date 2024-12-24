import time
import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import importlib

# download imagenet from huggingface https://huggingface.co/datasets/imagenet-1k
imgnet = importlib.import_module('imagenet-1k.classes')

import torchvision
torchvision.disable_beta_transforms_warning()

from data import dataset as ds
from torch.utils.data import DataLoader
from helpers import to_rgb, save_checkpoint, calculate_accuracy, validate_one_epoch, HGDataset
from torch.utils.tensorboard import SummaryWriter

from model import AlexNet


device = (
    "cuda" 
    if torch.cuda.is_available() 
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# tensorboard
writer = SummaryWriter()

# init logger
log = logger.init()

# hyperparameters
num_epochs = 90
batch_size = 128
initial_lr = 0.01
weight_decay = 0.0005
momentum = 0.9
num_classes = 1000
dropout = 0.5


# torch dataloaders
train_dataloader = DataLoader(ds('train'), batch_size=batch_size, shuffle=True,  num_workers=4)
test_dataloader  = DataLoader(ds('test'), batch_size=5, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(ds('validation'), batch_size=batch_size, shuffle=True, num_workers=4)

model = AlexNet(dropout, num_classes).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum)
train_size = len(train_dataloader.dataset)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True)

