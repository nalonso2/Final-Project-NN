#Convolutional Trainer


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
import convNN1



#Create vids
fHeight = 30
vids = vidGen2.motionVideoDataset(1500, fHeight, fHeight)



#Create model, loss function, and optimizer
predUnit = convNN1.predConvUnit().cpu()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    predUnit.parameters(), lr=1e-3, weight_decay=1e-5)



#TRAIN
#Cycle through videos. Train. 
losses = []
for vidNum in range(1,1500):
	[w,l,d] = vids[vidNum].size()
	vid = vids[vidNum]
	avgVidLoss = 0.0

	#Set initial prediction to zero vector, i.e. no prediction.
	prediction = torch.rand(1,1,fHeight,fHeight)

	#Cycle through frames. Compute loss and update weights each frame.
	for n in range(0,d-1):

		#Extract current frame (t0) and next frame (t1).
		frame_t0 = vid[0:fHeight, 0:fHeight, n]
		frame_t1 = vid[0:fHeight, 0:fHeight, n+1]


        #Format images so batch size is in first position
		frame_t0 = frame_t0.view(1,1,fHeight,fHeight)
		frame_t1 = frame_t1.view(1,1,fHeight,fHeight)

        #Compute Error between current frame (t0) and top-down prediction generated t-1
		error = frame_t0 - prediction
		

        #Feed forward error, and compute loss
		[prediction, model] = predUnit(error.detach())
		loss = criterion(prediction, frame_t1)
		avgVidLoss += loss.detach().numpy()

        #Reset gradients to zero, compute gradients, update parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


	losses.append(avgVidLoss / (d-1))
	print('video{}'.format(vidNum), ": ", avgVidLoss / (d-1))


torch.save(predUnit, 'predConv.pt')

plt.plot(losses)
plt.show()