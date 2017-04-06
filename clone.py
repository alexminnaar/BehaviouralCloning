import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle
import matplotlib.pyplot as plt


# import data
lines = []

with open('/home/aminnaar/sdc_course/sim_data_10/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # steering_angle = float(line[-4])
        # if abs(steering_angle) > 0.1:
        #     for i in range(3):
        #         lines.append(line)
        # else:
        #     lines.append(line)
        lines.append(line)
        
# parse data
images = []
measurements = []

shuffle(lines)

for line in lines:
    centre_image = line[0]
    left_image = line[1]
    right_image = line[2]
    image = [cv2.imread(x) for x in [centre_image, left_image, right_image]]
    images.extend(image)
    correction = 0.2
    measurement = float(line[3])
    # correct steering angle based on camera position
    measurements.extend([measurement, measurement + correction, measurement - correction])

augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

print X_train.shape

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model2.h5')

exit()
