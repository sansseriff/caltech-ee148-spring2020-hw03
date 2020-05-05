from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
import pickle

with open("test_losses_one.txt", "rb") as fp:
    test_losses_one = pickle.load(fp)

with open("test_losses_half.txt", "rb") as fp:
    test_losses_half = pickle.load(fp)

with open("test_losses_quarter.txt", "rb") as fp:
    test_losses_quarter = pickle.load(fp)

with open("test_losses_eighth.txt", "rb") as fp:
    test_losses_eighth = pickle.load(fp)

with open("test_losses_sixteenth.txt", "rb") as fp:
    test_losses_sixteenth = pickle.load(fp)


with open("train_losses_one.txt", "rb") as fp:
    train_losses_one = pickle.load(fp)

with open("train_losses_half.txt", "rb") as fp:
    train_losses_half = pickle.load(fp)

with open("train_losses_quarter.txt", "rb") as fp:
    train_losses_quarter = pickle.load(fp)

with open("train_losses_eighth.txt", "rb") as fp:
    train_losses_eighth = pickle.load(fp)

with open("train_losses_sixteenth.txt", "rb") as fp:
    train_losses_sixteenth = pickle.load(fp)


val_finals = [test_losses_one[-1],test_losses_half[-1],test_losses_quarter[-1],test_losses_eighth[-1],test_losses_sixteenth[-1]]
train_finals = [train_losses_one[-1],train_losses_half[-1],train_losses_quarter[-1],train_losses_eighth[-1],train_losses_sixteenth[-1]]
test_finals = [0.0191, 5.9967, 8.2877, 33.5006, 34.1454]
x = [51005, 25502, 12751, 6375, 3187] #manually recorded from running main.py



fig, ax = plt.subplots(1)
ax.loglog(x, test_finals, '-o', label='Test Losses', markersize=2)
ax.loglog(x, train_finals, '-o', label='Train Losses', markersize=2)
plt.legend()
plt.title("Train and Test Loss for Decreasing Train Dataset")

