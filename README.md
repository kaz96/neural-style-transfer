# Neural Style Transfer
![Neural Style Transfer Demo](demo/demo.gif)
## What is this?
Neural style transfer is the process of taking two images, one content image and a style reference image, and combines them together such that the subsequent output image contains the content image but with the style of the reference image. For our purposes, the content image we use are the individual frames captured with a webcam. 
        
        
## How it works
Without going too deep on the maths, neural style transfer is performed by using a pre-trained convolutional neural network and defining new loss functions which aims to minimize the difference between the content, the style and the resulting output image.
