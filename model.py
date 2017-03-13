import csv

import cv2

import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
base_path = 'track1/1'
with open(base_path + '/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#################### example generator #######################
import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
    	batch_samples = samples[offset:offset+batch_size]

    	images = []
    	angles = []
    	for batch_sample in batch_samples:
    		name = './IMG/'+batch_sample[0].split('/')[-1]
    		center_image = cv2.imread(name)
    		center_angle = float(batch_sample[3])
    		images.append(center_image)
    		angles.append(center_angle)

            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
	input_shape=(ch, row, col),
	output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= /
	len(train_samples), validation_data=validation_generator, /
	nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

#################### example generator #######################


images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		fileName = source_path.split("\\")[-1]
		current_path = base_path + '/IMG/' + fileName
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		# create adjusted steering measurements for the side camera images
		correction = 0.02
		steering_left = measurement + correction
		steering_right = measurement - correction
		if i == 0:
			measurements.append(measurement)
		elif i == 1:
			measurements.append(steering_left)
		else:
			measurements.append(steering_right)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)

x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print("angle:", measurements[0])

print("image:", x_train[0].shape)
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)

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