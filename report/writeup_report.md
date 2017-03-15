**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/lane_1.jpg "Center lane"
[image2]: ./examples/lane_2.jpg "Center lane"
[image3]: ./examples/recover_1.jpg "Recovery Image"
[image4]: ./examples/recover_2.jpg "Recovery Image"
[image5]: ./examples/recover_3.jpg "Recovery Image"
[image6]: ./examples/curve_1.jpg "Curve"
[image7]: ./examples/curve_2.jpg "Curve"
[image8]: ./examples/left.jpg "Left"
[image9]: ./examples/center.jpg "Center"
[image10]: ./examples/right.jpg "Right"
[image11]: ./examples/flip_1.jpg "Flipped"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5, 3x3 filter sizes and depths between 24 and 64 (model.py lines 66-105) 
The first 5 layers are confolution layers with relu activation function to introduce nonlinearity. 

The data is normalized in the model using a Keras lambda layer (code line 69). 
The images are also cropped in the model using a Keras cropping layer (code line 71).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers at layer 2, 4, 6 and 7 in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

I tuned the epoch number from 5 to 3 so that it doesn't overfit. Batch size is 32.

I also tried out different correction angle for the side cameras. It turned out that 0.2 is performing pretty well.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and curves. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia self driving car model. I thought this model might be appropriate because the training data and purpose would be very similar.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set at a rate of 80-20. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added drop out layers of rate 25% to 50% to some of the layers. I also shuffle the samples each time so that they are in random order. I changed the epoch number from 5 to 3. 

Then I started to use the left and right side camera pictures. However they were performing worse than just using the center camera. I found out that the correctio angle was too small that they polluted the overal trainning data. I tuned the angle to (-)0.2 in the end, which had a much better performance.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded more training data on the places where the vihecle fell off, especially when I'm getting close to fall off I recover from the edges.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 66-105) consisted of a convolution neural network with the following layers and layer sizes:
layer 1: convolution with kernel size 5x5, depth 24, stride 2x2, relu activation function
layer 2: convolution with kernel size 5x5, depth 36, stride 2x2, relu activation function, drop out 25%
layer 3: convolution with kernel size 5x5, depth 48, stride 2x2, relu activation function
layer 4: convolution with kernel size 3x3, depth 64, relu activation function, drop out 25%
layer 5: convolution with kernel size 3x3, depth 64, relu activation function
flatten 64x3x33 to 6336
layer 6: fully connected layer, 6336 -> 600, drop out 50%
layer 7: fully connected layer, 600 -> 100
layer 8: fully connected layer, 100 -> 10, drop out 25%
layer 9: output, 10 -> 1

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here are two examples of center lane driving:

![alt text][image1]
![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the lane once it's off. These images show what a recovery looks like starting from almost hitting the right side of the bride to be back to the center :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I also recorded a lot of curves so that the model would not be biased to only drive straight. Here are some examples of the curves:

![alt text][image6]
![alt text][image7]

I also used the left camera pictures and the right camera pictures with corrected steering angles. Here's a set of left, center and right images:

![alt text][image8]
![alt text][image9]
![alt text][image10]

To augment the data sat, I also flipped images and angles thinking that this would help to generalize the data. For example, here is an image that has then been flipped:

![alt text][image11]

I also drove in the opposite direction for one lap to generalize the data.


After the collection process, I had 52566 number of data points. I then preprocessed this data by normalize it and cropping the top 60 pixels and bottom 20 pixels from the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
