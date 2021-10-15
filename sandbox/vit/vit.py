import os

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets

#=============================================================================#
#                                     Data                                    #
#=============================================================================#

# Download data.
dpath = os.environ['HOME'] + '/.Data'

# TODO: should maybe also normalize?
transforms = T.ToTensor()
cifar10_train = datasets.CIFAR10(dpath, transform=transforms, 
								 download=True)
cifar10_test  = datasets.CIFAR10(dpath, transform=transforms, 
								 download=True, train=False)
b = 16
cifar10_train_dataloader = DataLoader(cifar10_train, batch_size=b)
cifar10_test_dataloader  = DataLoader(cifar10_test, batch_size=b)


#=============================================================================#
#                                     ViT                                     #
#=============================================================================#

