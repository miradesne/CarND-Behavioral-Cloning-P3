import csv

import cv2

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

samples = []
base_path = 'track1/3'

with open(base_path + '/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
				# use all pictures from the cameras
				for i in range(3):
					source_path = batch_sample[i]
					fileName = source_path.split("\\")[-1]
					current_path = base_path + '/IMG/' + fileName
					image = cv2.imread(current_path)
					images.append(image)
					measurement = float(batch_sample[3])
					# create adjusted steering measurements for the side camera images
					correction = 0.2
					steering_left = measurement + correction
					steering_right = measurement - correction
					if i == 0:
						measurements.append(measurement)
					elif i == 1:
						measurements.append(steering_left)
					else:
						measurements.append(steering_right)
			augmented_images, augmented_measurements = [], []
			# flip the images and measurements 
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image, 1))
				augmented_measurements.append(measurement*-1.0)
			x_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(x_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# create training model
model = Sequential()
# normalization
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
# cropping images to focus on the center part
model.add(Cropping2D(cropping=((60,20), (0,0))))

# layer 1: convolution 5x5x24, stride 2x2, relu activation function
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))

# layer 2: convolution 5x5x36, stride 2x2, relu activation function
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.25))

# layer 3: convolution 5x5x48, stride 2x2, relu activation function
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

# layer 4: convolution 3x3x64, relu activation function
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.25))

# layer 5: convolution 3x3x64, relu activation function
model.add(Convolution2D(64, 3, 3, activation='relu'))

# flatten 64x3x33 to 6336
model.add(Flatten()) 

# layer 6: fully connected layer, 6336 -> 600 
model.add(Dense(600))
model.add(Dropout(0.5))

# layer 7: fully connected layer, 600 -> 100
model.add(Dense(100))
model.add(Dropout(0.25))

# layer 8: fully connected layer, 100 -> 10
model.add(Dense(10))

# layer 9: output, 10 -> 1
model.add(Dense(1))

print("start training")
# mean square error with adam optimizer
model.compile(loss="mse", optimizer="adam")
history_object = model.fit_generator(train_generator, 
									samples_per_epoch=len(train_samples)*6, 
									validation_data=validation_generator, 
					        		nb_val_samples=len(validation_samples)*6, 
					        		nb_epoch=3)

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save("model.h5")