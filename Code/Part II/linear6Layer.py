#Final Project 3 Layer: Simple Predictive Coding Unit

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
class predLinearUnit(nn.Module):
    def __init__(self):
        super(predLinearUnit, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(900, 500),
            nn.ReLU(),
            nn.Linear(500, 364),
            nn.ReLU(),
        	nn.Linear(364, 300),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(300, 364),
            nn.ReLU(),
            nn.Linear(364, 500),
            nn.ReLU(),
            nn.Linear(500, 900),
            nn.Sigmoid())

    def forward(self,i):
    	model = self.encoder(i)
    	prediction = self.decoder(model)
    	return [prediction, model]


