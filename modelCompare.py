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
import stackedNN1
import finalProjVAE
import linear2Layer
import linear3Layer



#Create Videos
fHeight = 30
rSize = 144
rSize2 = 300
testVids = vidGen2.motionVideoDataset(500, fHeight, fHeight)




#load models. All models have same number of neurons: 3828

#hiearchy of 1 layer linear pred units. hidden at top of hiearchy is 300 neurons
ph = torch.load('predHierarchy.pt')

#Two layer linear pred unit. hidden layer is 500 neurons
l2 = torch.load('linear2Layer.pt')

#Three layer pred unit. hidden layer is 300 neurons
l3 = torch.load('linear3Layer.pt')



#COMPARE BCEs
phLoss = []
l2Loss = []
l3Loss = []


for vidNum in range(1, 500):
	phTotal = 0.0
	l2Total = 0.0
	l3Total = 0.0

	#Get video and video size d
	[w,l,d] = testVids[vidNum].size()
	vid = testVids[vidNum]


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



	#Cycle through frames. 
	for n in range(0, d):
		criterion = nn.BCELoss()
		
		#Extract current frame (t0).
		frame_t0 = vid[0:30, 0:30, n]
		frame_t0 = frame_t0.view(1,-1)

		#Combine partial img predictions from hierarchy to make full img prediction
		if n==0:
			imgPrediction = predL2
		else:
			imgPrediction = torch.cat((predsH[0], predsH[1], predsH[2], predsH[3]), 1)

		#Compute Errors for pred hierarchy and feedforward
		phTotal += criterion(imgPrediction, frame_t0)
		[errImg, errR, errTop] = ph.errSignal(frame_t0, repsH, predsH)
		[predsH, repsH] = ph.forward(errImg, errR, errTop)

		#Compute Errors for linear 2 layer predunit and feedforward
		l2Total += criterion(predL2, frame_t0)
		error2 = frame_t0 - predL2
		[predL2, rep2] = l2(error2)

		#Compute Errors for linear 3 layer predunit and feedforward
		l3Total += criterion(predL3, frame_t0)
		error3 = frame_t0 - predL3
		[predL3, rep3] = l3(error3)

	phLoss.append(phTotal.detach().numpy() / d)
	l2Loss.append(l2Total.detach().numpy() / d)
	l3Loss.append(l3Total.detach().numpy() / d)


#Plot losses
plt.plot(phLoss, label='LinearH')
plt.plot(l2Loss, label='Linear2')
plt.plot(l3Loss,  label='Linear3')
plt.legend()
plt.show()

print('LinearH Loss Avg: ', np.mean(phLoss), "std: ", np.std(phLoss))
print('Linear2 Loss Avg: ', np.mean(l2Loss),"std: ", np.std(l2Loss))
print('Linear3 Loss Avg: ', np.mean(l3Loss),"std: ", np.std(l3Loss))





#COMPARE PREDICTED IMAGES
for vidNum in range(1, 20):


	#Get video and video size d
	[w,l,d] = testVids[vidNum].size()
	vid = testVids[vidNum]


	


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


	#Create figure
	fig, ax = plt.subplots(4, d)
	ax[0,0].set(ylabel='Video')
	ax[1,0].set(ylabel='LinearH')
	ax[2,0].set(ylabel='Linear2')
	ax[3,0].set(ylabel='Linear3')

	#Cycle through frames. Plot each frame and prediction of frame
	for n in range(0, d):
		
		#Extract current frame (t0) and next frame (t1).
		frame_t0 = vid[0:30, 0:30, n]
		frame_t0 = frame_t0.view(1,-1)

		#Combine partial img predictions from hierarchy to make full img prediction
		if n==0:
			imgPrediction = predL2
		else:
			imgPrediction = torch.cat((predsH[0], predsH[1], predsH[2], predsH[3]), 1)


		#Plot current frame
		ax[0,n].imshow(frame_t0.view(fHeight,-1).detach())
		ax[1,n].imshow(imgPrediction.view(fHeight,-1).detach())
		ax[2,n].imshow(predL2.view(fHeight,-1).detach())
		ax[3,n].imshow(predL3.view(fHeight,-1).detach())


		#Compute Errors for pred hierarchy and feedforward
		[errImg, errR, errTop] = ph.errSignal(frame_t0, repsH, predsH)
		[predsH, repsH] = ph.forward(errImg, errR, errTop)

		#Compute Errors for linear 2 layer predunit and feedforward
		error2 = frame_t0 - predL2
		[predL2, rep2] = l2(error2)

		#Compute Errors for linear 3 layer predunit and feedforward
		error3 = frame_t0 - predL3
		[predL3, rep3] = l3(error3)


	
	plt.show()