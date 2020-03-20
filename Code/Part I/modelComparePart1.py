#Compare prediction hierarchy to 2 and 3 layer linear prediction units of same size


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
import finalProjVAE
import linear2Layer
import convNN1



#Create Videos
fHeight = 30
testVids = vidGen2.motionVideoDataset(500, fHeight, fHeight)




#load models. All models have same number of neurons: 3828

#hiearchy of 1 layer linear pred units. hidden at top of hiearchy is 300 neurons
l1 = torch.load('linear1Layer.pt')

#Two layer linear pred unit. hidden layer is 500 neurons
conv = torch.load('predConv.pt')

#Three layer pred unit. hidden layer is 300 neurons
vae = torch.load('predVAE.pt')



#COMPARE BCEs
l1Loss = []
convLoss = []
vaeLoss = []


for vidNum in range(1, 500):
	l1Total = 0.0
	convTotal = 0.0
	vaeTotal = 0.0

	#Get video and video size d
	[w,l,d] = testVids[vidNum].size()
	vid = testVids[vidNum]


	#Set initial predictions for linear predunits, i.e. no prediction.

	predL1 = torch.rand(1,900)
	predVAE = predL1
	predConv = torch.rand(1,1,fHeight,fHeight)

	#Cycle through frames. 
	for n in range(0, d):
		criterion = nn.BCELoss()
		
		#Extract current frame (t0) and flatten.
		frame_t0 = vid[0:30, 0:30, n]
		frame_t0 = frame_t0.view(1,-1)

	
		#Compute Errors for pred hierarchy and feedforward
		l1Total += criterion(predL1, frame_t0)
		error1 = frame_t0 - predL1
		[predL1, rep1] = l1(error1)

		#Compute Errors for linear 2 layer predunit and feedforward
		fConv = frame_t0.view(1,1,fHeight,-1)
		convTotal += criterion(predConv, fConv)
		error2 = fConv - predConv
		[predConv, rep2] = conv(error2)

		#Compute Errors for linear 3 layer predunit and feedforward
		vaeTotal += criterion(predVAE, frame_t0)
		error3 = frame_t0 - predVAE
		[predVAE, mu, var] = vae(error3)

	l1Loss.append(l1Total.detach().numpy() / d)
	convLoss.append(convTotal.detach().numpy() / d)
	vaeLoss.append(vaeTotal.detach().numpy() / d)


#Plot losses
plt.plot(l1Loss, label='Linear1')
plt.plot(convLoss, label='Conv')
plt.plot(vaeLoss,  label='VAE')
plt.legend()
plt.show()

print('Linear1 Loss Avg: ', np.mean(l1Loss), "std: ", np.std(l1Loss))
print('Conv Loss Avg: ', np.mean(convLoss),"std: ", np.std(convLoss))
print('VAE Loss Avg: ', np.mean(vaeLoss),"std: ", np.std(vaeLoss))





#COMPARE PREDICTED IMAGES
for vidNum in range(1, 20):


	#Get video and video size d
	[w,l,d] = testVids[vidNum].size()
	vid = testVids[vidNum]



	#Set initial predictions for linear predunits, i.e. no prediction.
	predL1 = torch.rand(1,900)
	predVAE = predL1
	predConv = torch.rand(1,1,fHeight,fHeight)


	#Create figure
	fig, ax = plt.subplots(4, d)
	ax[0,0].set(ylabel='Video')
	ax[1,0].set(ylabel='Linear1')
	ax[2,0].set(ylabel='Conv')
	ax[3,0].set(ylabel='VAE')

	#Cycle through frames. Plot each frame and prediction of frame
	for n in range(0, d):
		
		#Extract current frame (t0) and next frame (t1).
		frame_t0 = vid[0:30, 0:30, n]
		frame_t0 = frame_t0.view(1,-1)

	

		#Plot current frame
		ax[0,n].imshow(frame_t0.view(fHeight,-1).detach())
		ax[1,n].imshow(predL1.view(fHeight,-1).detach())
		ax[2,n].imshow(predConv.view(fHeight,-1).detach())
		ax[3,n].imshow(predVAE.view(fHeight,-1).detach())


		#Compute Errors for pred hierarchy and feedforward
		error1 = frame_t0 - predL1
		[predL1, rep1] = l1(error1)

		#Compute Errors for linear 2 layer predunit and feedforward
		fConv = frame_t0.view(1,1,fHeight,-1)
		error2 = fConv - predConv
		[predConv, rep2] = conv(error2)

		#Compute Errors for linear 3 layer predunit and feedforward
		error3 = frame_t0 - predVAE
		[predVAE, mu, var] = vae(error3)



	
	plt.show()