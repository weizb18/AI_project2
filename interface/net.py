import torch
import os
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet50
from torch.utils.data import DataLoader
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.linear_layer = nn.Linear(2048, 7)
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x