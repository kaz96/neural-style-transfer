# Neural Style Transfer
<p align="center">
  <img src="demo/demo.gif" alt="animated" />
</p>

## What is this?
Neural style transfer is the process of taking two images, one content image and a style reference image, and combines them together such that the subsequent output image contains the content image but with the style of the reference image. For our purposes, the content image we use are the individual frames captured with a webcam. 
        
        
## How it works
Neural style transfer is performed with a convolutional neural network that is already preatrained, typically using the ImageNet dataset. New loss functions are defined which aim to minimze the difference between the content image, the style image, and the resulting output image. 
