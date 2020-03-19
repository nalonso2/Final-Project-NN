#Final Project 2 Layer: Simple Predictive Coding Unit Trainer

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
import linear2Layer
import linear3Layer
import stackedNN1


#Create video dataset
fHeight = 30

vids = vidGen2.motionVideoDataset(1000, fHeight, fHeight)


#Create NNs, loss function, and optimizers
rSize = 144
rSize2 = 300
ph = stackedNN1.predHierarchy(fHeight, rSize, rSize2)

l2 = linear2Layer.predLinearUnit().cpu()
l2Opt = torch.optim.Adam(l2.parameters(), lr=7e-4)

l3 = linear3Layer.predLinearUnit().cpu()
l3Opt = torch.optim.Adam(l3.parameters(), lr=7e-4)

criterion = nn.BCELoss()




#TRAIN


def trainStepUnit(net, opt, error, img):
	criterion = nn.BCELoss()
	[prediction, model] = net(error.detach())
	loss = criterion(prediction, img)

	opt.zero_grad()
	loss.backward()
	opt.step()

	return [prediction, model, loss.detach().numpy()]





#Record losses
phLosses = []
l2Losses = []
l3Losses = []


#Cycle through videos. Train.
for vidNum in range(1,1000):

	#Extract frame and get size of vid d
	[w,l,d] = vids[vidNum].size()
	vid = vids[vidNum]


	#Reset the total loss per video
	phVidLoss = 0.0
	l2VidLoss = 0.0
	l3VidLoss = 0.0
	

	#Set initial predictions for hiearchy to rand vectors, i.e. no prediction.
	predsH = []
	repsH = []
	for i in range(0, 5):
		if i == 4: 
			predsH.append(torch.rand(1, int(4*rSize)))
			repsH.append(torch.rand(1, rSize2))
		else: 
			predsH.append(torch.rand(1, int((fHeight**2) / 4)))
			repsH.append(torch.rand(1, rSize))


	#Set initial predictions for linear predunits, i.e. no prediction.
	predL2 = torch.rand(1,900)
	predL3 = predL2

	#Cycle through frames. Compute loss and update weights each frame.
	for n in range(0,d-1):

		#Extract current frame (t0) and next frame (t1).
		frame_t0 = vid[0:30, 0:30, n]
		frame_t1 = vid[0:30, 0:30, n+1]

        #Flatten Images
		frame_t0 = frame_t0.view(1,-1)
		frame_t1 = frame_t1.view(1,-1)

		
        #Feed forward error, and compute loss for each nn
		[predsH, repsH, totLoss, imgLoss] = ph.trainStep(frame_t0, frame_t1, predsH, repsH)
		phVidLoss += imgLoss.detach().numpy()

		errorL2 = frame_t0 - predL2
		[predL2, repL2, loss] = trainStepUnit(l2, l2Opt, errorL2, frame_t1)
		l2VidLoss += loss

		errorL3 = frame_t0 - predL3
		[predL3, repL3, loss] = trainStepUnit(l3, l3Opt, errorL3, frame_t1)
		l3VidLoss += loss




	phLosses.append(phVidLoss / (d-1))
	l2Losses.append(l2VidLoss / (d-1))
	l3Losses.append(l3VidLoss / (d-1))
	print('video{}'.format(vidNum))


#Plot losses
plt.plot(phLosses, label='LinearH')
plt.plot(l2Losses, label='Linear2')
plt.plot(l3Losses,  label='Linear3')
plt.legend()
plt.show()
