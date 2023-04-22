# Link To The Video:
# https://www.youtube.com/watch?v=hvLFD4AZzCw&list=PLyMom0n-MBroupZiLfVSZqK5asX8KfoHL&index=3

import torch
import torchvision

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

# Working with images
dataset = MNIST(root='data/', download=True)
batch_size = 128

train_ds, val_ds = random_split(dataset, [50000, 10000])
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

