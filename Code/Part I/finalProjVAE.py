#Predictive Coding unit variational. 
#1. Note this VAE class is based off of code found here https://github.com/pytorch/examples/blob/master/vae/main.py
#2. The code was an implementation of Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
#3. (I used this code as the basis for my VAE because I had trouble accessing the VAE on canvas) 

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import vidGen2
import matplotlib.pyplot as plt




#Generate Data
fHeight = 30
trainVids = vidGen2.motionVideoDataset(1000, fHeight, fHeight)
testVids = vidGen2.motionVideoDataset(300, fHeight, fHeight)



#Variational autoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(900, 400)
        self.fc21 = nn.Linear(400, 120)
        self.fc22 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 400)
        self.fc4 = nn.Linear(400, 900)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 900))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar





