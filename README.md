# Simple Predictive Coding Networks for Shape and Motion Prediction
### By Nick Alonso

  Those building artificial neural networks for vision tasks, will often, maybe typically, utilize feedforward neural networks. Cortical regions of the human brain, especially the visual cortex, however, utilizes large amounts of feedback connections. There is good reason, then, to believe that neural networks may be improved and reach more human-like performance by utilizing feedback connections like the human brain does. The challenge, however, is figuring out how to integrate feedback connections into neural networks and figuring out how and what they should be trained to do. Currently, a leading theory of what feedback connections do in the brain is the theory of predictive coding. This theory states that the large number of feedback connections found in cortical regions of the brain are used, mainly, to predict neural activity at lower cortical regions (Spratling, 2017). Feedforward connections are tasked with propagating forward an error signal: a measure of the difference between predicted neural activity and actual activity. This architecture allows the brain to encode information efficientaly, as only information about unpredicted aspects of neural activity is propogated forward rather than information about all of it (Rao and Ballard, 2011). This theory was developed and modeled, most notabely, by Rao and Ballard (1999). They built a hierarchy of small autoencoder-like neural networks. The encoder of each unit of the hierarchy took in error signals as input and outputted a compressed representation of the input (r), then r was inputted into the decoder which attempted to generate (i.e. predict) the input at the next timestep. Each unit, specifically, attempted to predict r in a subset of the units at lower levels in the hierarchy (more details below).
  
  My goal for this project was to 1. develop a better understanding of how predictive coding models work (with a special focus on Rao and Ballard (1999)) and 2. to implement some simple predictive coding networks using pytorch. My focus was to learn and implement a working predictive coding architecture rather than to build a network for the purpose of meeting or exceeding some performance benchmark. For this reason, I decided to build a larger number of small neural networks that trained quickly on somewhat simple data, rather than one or a few large networks on more complex data. This allowed me to experiment with various predictive coding architectures quickly, and understand how they worked and what might improve (or worsen) their performance.
  
  In this paper, I present some of the findings of this project. I divide my findings into two parts. In the first part, I compare three simple predictive coding units I built (PCUs): one with linear layers, one with convolutional layers, and a variational autoencoder. In the second part, I describe how I combined five simple linear PCUs into a hierarchy, and I compare the performance of this hierarchy to PCUs that are not hierarchical but are of similar size.
  
## 1. Data: Video Generation
  Rao and Ballard (1999) built a predictive coding unit to generate an image, but the same unit could also be used to predict video. In the case of video prediction, the unit would not simply be trained to generate (at time t1) the same image presented at a prior timestep t0), but rather to predict what new image of the video will appear (at time t1) given the image that appeared at a prior timestep (time t0). In order to predict video, the unit has to learn regularities in the way images change over time rather than just the spatial properties of a single image. Predictive coding inspired neural networks have been applied by others for the purpose of video prediction with some success (e.g. .....).
  
  I decided to use video as my input for my networks. My neural networks were going to be small, so the videos used as input could not be too complex. I decided generating simple video would be the easiest way to get this data. I created videos of black rectangles moving across a white background. The size of the background could be adjusted, but I used a small 30x30 frame for all of my tests. Each rectangle was composed of roughly the same number of pixels (12 to 16), but had one of five different shapes: 4x4, 3x5, 5x3, 2x8, 8x2. Each rectangle moved either straight up or down, straight left or right, or at a 45 degree angle up-left, up-right, down-left, or down-right. They also moved at various speeds (usually 1-4 pixels per frame). Thus, the neural networks, in order to generate the videos accurately, had to learn to encode information about shape and motion (i.e. direction and speed).
  
  Each video started with a rectangle positioned randomly somewhere around the center, then proceeded to move until it left the screen. Each video was a 4D pytorch tensor, (batch, frameHeight, frameWidth, frameNumber). These were loaded into a dictionary. Using a dictionary allowed me to store videos of varying length in the same data structure. This would not be possible if I had instead padded the videos to all be the same length and stored them in a tensor. I tried the padding technique, but it affecting the results (likely because many videos were quite short and needed a lot of padding). So I decided to not use padding, but instead store the videos in a dictionary without padding.

## 3. PartI: Three Simple Predictive Coding Units

  The first part of my project consisted of building and experimenting with single predictive coding units with various kinds of layers, activation functions, and size. I present three of these units here. Each predictive coding unit can be seen as special kind of autoencoder. It is important, first, to keep in mind the predictive coding unit is not just generating an image, but is predicting a future image in a video. Second, the encoder of the PCU does not take in an image as input, but rather takes in an error signal as input, which I will explain below. Third, predictive coding units can be combined into a hiearchy of units and take in input from units higher-up in the hierarchy. I focus on building hiearchy in the next section. Here I will only focus on building a single predictive coding unit.
  
  An error signal can be understood as follows. Let's call the image at timestep zero I<sub>0</sub> and the prediction of that image generated by a predictive coding unit (PCU) I<sub>Pred0</sub>. The error signal that the PCU will take as input to predict the next frame at time one is some measure the different between I<sub>0</sub> and I<sub>Pred0</sub>. Rao and Ballard used the simple measure I<sub>0</sub> - I<sub>0</sub>, where the subtraction is an element-wise subtraction of the two matrices. The result is a error signal which is then inputted to the PCU. Other measures of the error may be used. Spratling (2017), for example, suggests that a*I<sub>0</sub> / b*I<sub>0</sub> + c is more biologically plausible (a,b, and c are constants). (Though I don't show results, I found both errors acheived similar results so I stuck with Rao and Ballards simpler error computation). Rao and Ballard (1999, 2011) argue the error signal is a more efficient way to encode and feedforward information, as only information about the unpredicted/unexpected aspects of the input are propogated forward rather than all of it, which, on average, reduced feedforward activity.

### 3.1 Building Predictive Coding Units
I experimented with many different kinds of PCU. Here I present three simple ones. First, is a PCU that consists of two linear layers (not counting the input layer), a one layer encoder and a one layer decoder. The input and output layer are size 1x900. The hidden layer is of size 1x400. The encoder used ReLU activation functions, while the decoder used sigmoid, as its ouput must be between 0 and 1.

Second, is a  PCU that used convolutional layers. I consisted of two layers (not including input layer), 1 convolutional layer encoder and one convolutional layer decoder. Input and output layers were 1x30x30. A kernal of size 4 and a stride of two were ysed, with 1 to 3 output channels. Here I show the results of a PCU with an encoder that has 2 output channels. 

Third, I created a variational autoencoder. I used code found here https://github.com/pytorch/examples/blob/master/vae/main.py to build the network. This code is an implementation of Kingma and Wellington (2014). This network had an input layer of size 1x900, which fed into a hidden layer h1 of 1x400. H1 fed into two more hidden layers of size 1x120. These encoded mu and var variables (see code). Mu and var are used for reparameterization, produces a vector of size 1x120. This is fed into the decoder which consists of three linear layers size, 1x120, 1x400, and 1x900.

### 3.2 Training and Results
Each network took an error signal as input. The error signal was, as noted above, I<sub>0</sub> and I<sub>Pred0</sub>. They were trained using backpropigation. Here I used BCE loss to train the linear and convolutional PCU units. (Rao and Ballard (1999) use a different optimization function but I do not use that here). The loss function for the variational autoencoder was more complex, and I will refer you to the code (found in finalProjVAE.py) for more details. The convolutional and linear neural net were trained over 1500 video. The variational autoencoder too many more videos, however, to acheive a decent performance. It was trained on 750 videos of 17 epochs.

Each network performed slightly differently. Below you will see a comparison of their BCE loss over the same 500 test videos. The BCE was average each video


Below you will see a side by side comparison of video frames and the predictions of those frames produces by each PCU for one video. Each network predicts random noise on the first frame. Then they all roughly generate the previous frame. One frame is not enough to know what direct and speed these shapes are moving yet, so they will need one or more frames to figure out this information.

It becomes clear that both the VAE and the linear PCU predict motion, as they eventually stop generating the previous frame and begin predicting that the shape will be in a different position than it was last frame (i.e. that it will move). The VAE, however, seems better than the simple linear unit at encoding shape. The convolutional PCU generates shape well, but never seems to encode information about motion. It always seems to generate the previous frame. This may be because the PCU is so small and has so few channels and layers. (I did attempt to create larger convolutional PCUs, but could not improve my results. I am still trying to figure out why this is the case).

## 4. PartII: Hierarchy of Predictive Coding Units
  Next, I created a hierarchy of PCUs. Rao and Ballard (1999) argued that the cortical areas of the brain can be seen as hierarchies of predictive coding units. Each unit attempts to predict the neural activity of the hidden layers/representations of PCUs at lower levels in the hierarchy. An error signal is propagated forward based on the difference between the predicted neural activity and the actual neural activity. In addition to the bottom up error signal, the activity in PCU hidden layers is influenced by the top down prediction of the hidden layer. Rao and Ballard (1999) implemented this top-down influence by taking the error between the neural activity of a PCU's hidden layer (r) and its top-down prediction: r - r<sub>topdown</sub>. Then they inputted this top-down error into the encoder of the PCU, so now the encoder takes as input the bottom-up error (the error in its prediction of a lower unit's r) and the top down error (a higher units prediction of its r) and outputs a compressed representation r of lower level neural activity. 

### 4.1 Building PCU Hierarchy
  I implemented this same kind of hierarchy using five linear PCU's of the sort described in Part I. The hierarchy had two levels. The first level consisted of four PCU's, each was responsible for predicting one quarter of the image pixels. The second level consisted of one PCU that predicted the hidden layers of the four lower PCUs. The lower unit's encoder took as input the bottom up error signal (which was the difference between its prediction of 1/4 of the image pixels and the actual pixels of that image patch), and it took as input the error between its hidden layer r and the prediction of r coming from the higher level unit. Because there was no third level, the second level PCU's hidden layer was not influenced by top-down predictions.
  
  The video images were 30x30. The encoders of the four lower level PCU's took as input 225 image pixels. Their hidden layers were of size 144. Thus, the top-down error signal is of size 144 making their encoder's total input layer 1 x (225 + 144) = 1 x 369. The output layers were 1 x 225 (i.e. the size of the image patch they are predicting). The second layer PCU took in all the hidder layers (the rs) of the four lower level PCUs as input. Thus, its input (and output) layer was 1 x 576. Its hidden layer was of size 1 x 300. All layers used sigmoid activation functions.
  
 Each unit was trained using its own adam optimizer. A BCE loss function was used. For lower units, the loss was computed between the image patch they were predicting and their prediction of it. The second level PCU's loss was computed between the values of the hidden layers of the lower units and its prediction of those values.
 
 I compared the performance of the hierarchy to two other neural networks. Each is a single PCU that has the same number of neurons as the hierarchy does (i.e. 3828 neurons). One neural network has 4 linear layers (not including input layer) with hidden layer of 1 x 500. The second has 6 linear layers (not including input layer) with a hidden layer of size 1 x 300. All layers in the networks use ReLU activation functions, except for the output layers, which use sigmoid (these networks would not work, unlike the hierarchy, using only sigmoid activations). Despite have the same number of neurons, it should be noted the hierachy has fewer connections because the hiden layers of the lower PCUs are not fully connects to those below it. Specifically, the hidden layers at the lower levels are only connected to one fourth of the input neurons (i.e. those neurons in the same PCU) rather than all of them, which is not the case with these other two neural networks which are fully connected. The six layer PCU also has more layers than the hierachy (6 compared to 4), and the four layer PCU compresses the input less than the hierarchy: the entire image is compressed to 300 neurons in the hierarchy but only 500 in the four layer PCU. The reason for comparing these neural networks was to see if the hierarchy had any advantages in performance over neural networks of similar size. 
 
 ### 4.2 Results
 
Results can be seen below. LinearH is the hiearchy of PCUs. Linear2 is the four layer PCU and Linear3 is the six layer PCU. The first graph shows their avg BCE loss per video during training across 300 videos (i.e. average BCE loss between entire image and prediction of full image). The rate of improvement is quite similar for the hierarchy as it is for the single prediction units. Though near the end the is some divergence in performance as they level out, which I measure below.
 
 
In the graph below, I tested each neural network's avg BCE loss per video after it had been trained across 500 hundred test videos. Each neural net was trained on the same 1000 training videos. Although these performances look similar. There are some important differences. First, the hierachy's performance has greater variance across videos than do the other neural networks. Second, on average, the hierarchy has lower average BCEs than the other two neural networks. Over these 500 videos the mean of the average BCE Loss were as follows: LinearH = .322  Linear2 = .328 Linear3 = .332. The standard deviations were: LinearH = .116  Linear2 = .107 Linear3 = .099. These may seem like small differences, but they are consistent. Across 10 more runs of different sets of 500 videos, LinearH always had the smallest mean BCE and the largest std. Additionally, I suspect that if the input  was made to be more complex, it would yield generally larger losses and greater divergences between the performances of these models, though this would have to be tested to be confirmed.

Comparing the predictions of each neural network makes the performance differences more clear. While all the models seem to predict motion relatively well, the hierarchy is clearly encoding and predicting shape better than the other two models. In the image comparison below, for example, the hierarchy is clearly encoding the tall skinny rectangle shape much better than the other neural nets. Just about every visual comparison shows similar results (see Data/PartII for more results). The PCUs seem only able to generate elongated ovals with fuzzy boundaries, will the hierarchy is able to generate more rectangular shapes with sharper edges.


## 5. Discussion

