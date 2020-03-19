#Simple Stacked Prediction Units

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



#Create predictive linear unit. 
#io is input/output size, r is representation/hidden layer size.
class predLinearUnit(nn.Module):

    def __init__(self, io, r, top=False):
        super(predLinearUnit, self).__init__()

        #Compute input size. If unit is not on top level of hierarchy, 
        #then input will include top down error signal, which is same size as r.
        self.top = top
        self.inputSize = io if top else (io+r)

        #Create encoder decoder
        self.encoder = nn.Sequential(
            nn.Linear(int(self.inputSize), r),
            nn.Sigmoid()) 
        self.decoder = nn.Sequential(
            nn.Linear(r, io),
            nn.Sigmoid())


   
    def forward(self, inpt, errorTopDown=None):

    	#If not top unit, add top down error to input
    	if not self.top:
    		inpt = torch.cat((inpt, errorTopDown), 1)

    	#Feedforward
    	r = self.encoder(inpt)
    	pred = self.decoder(r)
    	return [pred, r]




#Predictive coding hierachy. Two levels. 
#Four predUnits on lower level, each takes in 1/4 of image.
#One unit on top level. Takes in r (the hidden layer) from all lower level units.
class predHierarchy():


	def __init__(self, imageHeight, r, r2):

		#Compute input sizes for each layer of pred units (low and high)
		self.inputSizeLow = imageHeight*imageHeight / 4
		self.inputSizeHigh = 4*r
		self.r = r
		self.r2 = r2

		#Create lower level pred units and their optimizers. Store in list
		self.pUnitsLow =[]
		self.optsLow = []

		for i in range(0, 4):
			self.pUnitsLow.append(predLinearUnit(int(self.inputSizeLow), r))
			self.optsLow.append(torch.optim.Adam(self.pUnitsLow[i].parameters(), lr=7e-4))


		#Create top predunit and optimizer.
		self.pUnitTop = predLinearUnit(self.inputSizeHigh, r2, True)
		self.optTop = torch.optim.Adam(self.pUnitTop.parameters(), lr=7e-4)




	#This function computes error signals that will be fed into each pUnit
	def errSignal(self, image_t0, Rs, predictions):

		#errImg are errors in predicted image values made by lower units at t0. 
		errImg = []
		for x in range(0, 4):
			errImg.append(image_t0[0, (int(x * self.inputSizeLow)): (int((x+1) * self.inputSizeLow))] - predictions[x])

		#errR are errors in top unit's predictions of lower units' r values at t0. Each errR is fed into a lower unit
		errR = []
		for x in range(0, 4):
			errR.append(Rs[x] - predictions[4][0, int(x * self.r): int((x+1) * self.r)])


		#errTop is the error signal fed into top unit. It is concatiation of all errRs
		errTop = torch.cat((errR[0], errR[1], errR[2], errR[3]), 1)

		return [errImg, errR, errTop]



	#Feed forward error signals through each predUnit and output new predictions and r values
	def forward(self, errImg, errR, errTop):
		#Low level
		newPreds = []
		newRs = []
		for i in range(0,4):
			[pred, r] = self.pUnitsLow[i](errImg[i].detach(), errR[i].detach())
			newPreds.append(pred)
			newRs.append(r)

		#Top Level
		[pred, r] = self.pUnitTop(errTop.clone().detach())
		newPreds.append(pred)
		newRs.append(r)

		return [newPreds, newRs]



	#Compute BCE loss for each predunit. Perform backprop on each predunit.
	#Output combined BCE across predunits and the bce loss of the image and its full prediction.
	def train(self, newPreds, newRs, img):
		criterion = nn.BCELoss()
		totalLoss = 0

		#Train lower level units
		for i in range(0, 4):
			loss = criterion(newPreds[i], img[0:1, int((i*self.inputSizeLow)): int(((i+1) * self.inputSizeLow))])
			self.optsLow[i].zero_grad()
			loss.backward()
			self.optsLow[i].step()
			totalLoss += loss.detach().numpy()

		#Train top Level unit
		rAll = torch.cat((newRs[0], newRs[1], newRs[2], newRs[3]), 1)
		loss = criterion(newPreds[4], rAll.detach())
		self.optTop.zero_grad()
		loss.backward()
		self.optTop.step()
		totalLoss += loss.detach().numpy()

		#Find loss of the prediction of full image.
		imgPrediction = torch.cat((newPreds[0], newPreds[1], newPreds[2], newPreds[3]), 1)
		imgLoss = criterion(imgPrediction, img)


		return [totalLoss, imgLoss]



	#Input is current t0 (flattened) image, next t1 (flattened) image, and predictions of t0 generated at t-1.
	#Outputs are new predictions, representations, and combined BCE loss across predunits.
	def trainStep(self, image_t0, image_t1, predictions, Rs):

		#Compute Error Signal
		[errImg, errR, errTop] = self.errSignal(image_t0, Rs, predictions)

		#Feedforward Error Signal. Collect new predictions and represenations
		[newPreds, newRs] = self.forward(errImg, errR, errTop)

		#Compute BCE Loss of new preds at t1 and Backpropogate
		[totLoss, imgLoss] = self.train(newPreds, newRs, image_t1)

		return [newPreds, newRs, totLoss, imgLoss]






