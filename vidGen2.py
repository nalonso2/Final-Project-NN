#Creates videos of variable length stored in dictionary




import os
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import random
import matplotlib.pyplot as plt
import numpy as np


class motionVideoDataset(Dataset):

	def __init__(self, numVids, frameHeight, frameWidth, minSpeed=1, maxSpeed=4):
		
		self.numberVids = numVids
		self.frameHeight = frameHeight
		self.frameWidth = frameWidth
		self.vids = {}


		#First create 5 different rectangles represented a [x,y] (width and height), different directions, and speeds
		shapes = [[3, 5], [4, 4],  [5, 3], [2, 8], [8, 2]]
		directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
		speed = list(range(minSpeed, maxSpeed+1))

		
		#Create videos.
		vidNum = 0
		for x in range(self.numberVids):
			
			vidNum+=1
						
			#Set starting position, shape, direction, speed, and create new vid.
			position = [random.randint(7, frameWidth-6), random.randint(7,frameWidth-6)]
			s = random.choice(shapes)
			d = random.choice(directions)
			v = random.choice(speed)
			vid = torch.zeros(self.frameHeight, self.frameWidth, 0)

			#Add frames to video until center of shape leaves frame
			while ((position[0] < self.frameWidth) & (position[1] < self.frameHeight) & (position[0] > 0) & (position[1] > 0)):

				#First, create new frame of white background
				frame = torch.zeros(self.frameHeight, self.frameWidth, 1) + .01

				#Find indeces for each side of rectangle. [right, left, top, bottom]
				indeces = [position[0] - s[0], position[0] + s[0], position[1] - s[1], position[1] + s[1]]
				
				#Check if shape is outside of left or right of frame
				for y in range(2):
					if indeces[y] < 0: indeces[y] = 0
					if indeces[y] > self.frameWidth: indeces[y] = self.frameWidth

				#Check if shape is outside of top or bottom of frame
				for y in range(2,4):
					if indeces[y] < 0: indeces[y] = 0
					if indeces[y] > self.frameHeight: indeces[y] = self.frameHeight

				#Add black shape to white frame using indeces
				frame[indeces[0]:indeces[1], indeces[2]:indeces[3], 0] = .99
					
				#Add frame to vid
				vid = torch.cat((vid, frame), 2)

				#Update x and y position
				position[0] += d[0] * v
				position[1] += d[1] * v
					

			#Load vid into the vid dictionary
			self.vids['Video{}'.format(vidNum)] = vid
		



	def __len__(self):
		return len(self.vids)



	def __getitem__(self, idx):
		return self.vids['Video{}'.format(idx)]



	def playVideos(self, playN):
		#Play videos
		for x in range(1,playN):
			[w,l,d] = self.vids[x].size()
			for n in range(0,d):
				plt.imshow(self.vids[x][0:30, 0:30, n])
				plt.show(block=False)
				plt.pause(.5)
				plt.close()




#TEST
#vids = motionVideoDataset(1000, 30, 30)


#vids.playVideos(5)