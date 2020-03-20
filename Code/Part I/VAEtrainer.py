#Trainer for VAE predUnit


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import vidGen2
import matplotlib.pyplot as plt
import finalProjVAE


#Generate Data
fHeight = 30
trainVids = vidGen2.motionVideoDataset(750, fHeight, fHeight)


#Create neural network and optimizer
predUnit = finalProjVAE.VAE().cpu()
optimizer = optim.Adam(predUnit.parameters(), lr=1e-3)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 900), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD



#TRAIN
losses = []
for epoch in range(0, 17):
	avgVidLoss = 0.0

	#Cycle through videos
	for vidNum in range(1, 750):

		#Get video and video size d
		[w,l,d] = trainVids[vidNum].size()
		vid = trainVids[vidNum]
		

		#Set initial prediction to rand vector, i.e. no prediction.
		prediction = torch.rand(1, fHeight*fHeight)

		#Cycle through frames. Compute loss and update weights each frame.
		for n in range(0, d-1):

			#Extract current frame (t0) and next frame (t1).
			frame_t0 = vid[0:30, 0:30, n]
			frame_t1 = vid[0:30, 0:30, n+1]

			#Flatten Images
			frame_t0 = frame_t0.view(1,-1)
			frame_t1 = frame_t1.view(1,-1)

		    #Compute Error between current frame (t0) and top-down prediction generated t-1
			error = frame_t0 - prediction
			
		    #Feed forward error, and compute loss
			prediction, mu, logvar = predUnit(error.detach())
			loss = loss_function(prediction, frame_t1, mu, logvar)
			avgVidLoss += loss

		    #Reset gradients to zero
			optimizer.zero_grad()

			#Compute accumulated gradients
			loss.backward()

			#Perform parameter update based on current gradients
			optimizer.step()


		losses.append(loss.detach().numpy())
	print('Epoch{}'.format(epoch), ": ", avgVidLoss / 750)

#Plot losses
plt.plot(losses)
plt.show()


torch.save(predUnit, 'predVAE.pt')

