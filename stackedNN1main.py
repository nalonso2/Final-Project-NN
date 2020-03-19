

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





#Create Videos
fHeight = 30
trainVids = vidGen2.motionVideoDataset(750, fHeight, fHeight)


#Create predictive unit hierarchy
rSize = 144
rSize2 = 300
ph = stackedNN1.predHierarchy(fHeight, rSize, rSize2)




#Train hierarchy
totalLosses = []
imgLosses = []
for epoch in range(0, 7):

	#Cycle through videos
	for vidNum in range(1, 750):
		avgImgLoss = 0.0

		#Get video and video size d
		[w,l,d] = trainVids[vidNum].size()
		vid = trainVids[vidNum]


		#Set initial predictions to rand vectors, i.e. no prediction.
		preds = []
		reps = []
		for i in range(0, 5):
			if i == 4: 
				preds.append(torch.rand(1, int(4*rSize)))
				reps.append(torch.rand(1, rSize2))
			else: 
				preds.append(torch.rand(1, int((fHeight**2) / 4)))
				reps.append(torch.rand(1, rSize))



		#Cycle through frames. Compute loss and backprop each frame.
		for n in range(0, d-1):

			#Extract current frame (t0) and next frame (t1).
			frame_t0 = vid[0:30, 0:30, n]
			frame_t1 = vid[0:30, 0:30, n+1]

			frame_t0 = frame_t0.view(1,-1)
			frame_t1 = frame_t1.view(1,-1)

			[preds, reps, totLoss, imgLoss] = ph.trainStep(frame_t0, frame_t1, preds, reps)
			avgImgLoss += imgLoss


		#Record losses (totLoss is summed BCE loss of all predunits in hierarchy at a 
		#timestep. imgLoss is bce of full image prediction at a timestep)
		totalLosses.append(totLoss)
		imgLosses.append(avgImgLoss / (d-1))
		print("Epoch: ", epoch, "Video: ", vidNum, " Loss: ", totLoss)






#Plot losses
plt.plot(totalLosses)
plt.show()
plt.plot(imgLosses)
plt.show()




#TEST/SHOW
#Cycle through videos
for vidNum in range(1, 20):


	#Get video and video size d
	[w,l,d] = testVids[vidNum].size()
	vid = testVids[vidNum]


	#Create figure
	fig = plt.figure()
	height = 2
	width = d


	#Set initial predictions to rand vectors, i.e. no prediction.
	preds = []
	reps = []
	for i in range(0, 5):
		if i == 4: 
			preds.append(torch.rand(1, int(4*rSize)))
			reps.append(torch.rand(1, rSize2))
		else: 
			preds.append(torch.rand(1, int((fHeight**2) / 4)))
			reps.append(torch.rand(1, rSize))



	#Cycle through frames. Plot each frame and prediction of frame
	for n in range(0, d):

		#Extract current frame (t0) and next frame (t1).
		frame_t0 = vid[0:30, 0:30, n]
		frame_t0 = frame_t0.view(1,-1)

		#Combine predictions to make image
		imgPrediction = torch.cat((preds[0], preds[1], preds[2], preds[3]), 1)

		#Plot current frame
		fig.add_subplot(height, width, n+1)
		plt.imshow(frame_t0.view(fHeight,-1).detach())

		#Below current frame, plot prediction of current frame
		fig.add_subplot(height, width, (n+1) + d)
		plt.imshow(imgPrediction.view(fHeight,-1).detach())

		#Compute Error Signals
		[errImg, errR, errTop] = ph.errSignal(frame_t0, reps, preds)

		#Feedforward Error Signals. Collect new predictions and representations
		[preds, reps] = ph.forward(errImg, errR, errTop)

	plt.show()


torch.save(ph, 'predHierarchy3.pt')