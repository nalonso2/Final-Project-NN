# Simple Predictive Coding Networks for Representing Shape and Motion
### By Nick Alonso

## 1. Introduction
  1. intro to topic
  2. intro to predictive coding.
  3. My goal for this project was to 1. develop a better understanding of how  predictive coding models work (with a special focus on Rao and Ballard (1999)) and 2. to implement some simple predictive coding networks using pytorch. My focus, therefore, was to learn and implement a working predictive coding architecture rather than to build a network for the purpose of meeting or exceeding some performance benchmark. For this reason, I decided to build a larger number of small neural networks that trained quickly on somewhat simple data, rather than one or a few large networks on more complex data. This allowed me to experiment with various predictive coding architectures quickly, and understand how they worked and what might improve (or worsen) their performance.
  4. 
  
## 2. Data: Video Generation
  Rao and Ballard (1999) built a predictive coding unit to generate an image, but the same unit could also be used to predict video. In the case of video prediction, the unit would not simply be trained to generate (at time t1) an image (presented at prior timestep t0), but rather to predict what new image of the video will appear (at time t1) given the image that appeared at a prior timestep (time t0). In order to predict video, the unit has to learn regularities in the way images change over time rather than just the spatial properties of the image. Predictive coding units have been applied by others for the purpose of video prediction with some success (e.g. .....).
  
  I decided to use video as my input for my networks. My neural networks were going to be small, so the videos used as input could not be too complex. I decided generating simple video would be the easiest way to get this data. I created videos of black rectangles moving across a white background. The size of the background could be adjusted, but I used a small 30x30 frame for all of my tests. Each rectangle was composed of roughly the same number of pixels (12 to 16), but had one of five different shapes: 4x4, 3x5, 5x3, 2x8, 8x2. Each rectangle moved either straight up or down, straight left or right, or at a 45 degree angle up-left, up-right, down-left, or down-right. They also moved at various speeds (usually 1-4 pixels per frame). Thus, the neural networks, in order to generate the videos accurately, had to learn to encode information about shape and motion (i.e. direction and speed).
  
  Each video started with a rectangle positioned randomly somewhere around the center, then proceeded to move until it left the screen. Each video was a 4D pytorch tensor, (batch, frameHeight, frameWidth, frameNumber). These were loaded into a dictionary. Using a dictionary allowed me to store videos of varying length in the same data structure. This would not be possible if I had instead padded the videos to all be the same length and stored them in a tensor. I tried the padding technique, but it affecting the results (likely because many videos were quite short and needed a lot of padding). So I decided to not use padding, but instead store the videos in a dictionary without padding.

## 3. PartI: Three Simple Predictive Coding Units
  The first part of my project consisted of building and experimenting with single predictive coding units with various kinds of layers, activation functions, and size. I present three of these units here. Each predictive coding unit can be seen as special kind of autoencoder. First, it is important to keep in mind the autoencoder is not just generating an image, but is predicting a future image. Second, the encoder does not take in an image as input, but rather takes in an error signal as input, which I will explain below. Third, predictive coding units can be combined into a hiearchy of units and take in input for units higher-up in the hierarchy. I focus on building hiearchy in the next section. Here I will only focus on building single predictive coding units.
  Let's call the image at a timestep 0 I<sub>subscript</sub>



## 4. PartII: Hierarchy of Predictive Coding Units


### 4.1 Building

## 5. Conclusion
