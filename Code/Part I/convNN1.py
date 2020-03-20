#Final Project Convlayer: Simple Predictive Coding Unit

import os
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import random
import matplotlib.pyplot as plt
import numpy as np
import vidGen2




#Create Neural Network
class predConvUnit(nn.Module):
    def __init__(self):
        super(predConvUnit, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4, stride=2),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2),
            nn.Sigmoid())

    def forward(self, i):
    	model = self.encoder(i)
    	prediction = self.decoder(model)
    	return [prediction, model]

