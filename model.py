import csv
import cv2
import numpy as np
import sklearn

samples = []
with open('./Collected_Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset: offset+batch_size]
			
			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					name = './Collected_Data/IMG/' + batch_sample[i].split('\\')[-1]
					image = cv2.imread(name)
					angle = float(batch_sample[3])
					images.append(image)
					images.append(cv2.flip(image, 1))
					if i == 0:
						angle = angle
					elif i == 1:
						angle = angle + 0.2
					else:
						angle = angle - 0.2
					angles.append(angle)
					angles.append(angle*-1.0)
			
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
		
#Complie and train the model using generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
import matplotlib.pyplot as plt

#Build NN model	
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25),(0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#model.summary()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = 
    len(train_samples), validation_data = 
    validation_generator, 
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)
	
model.save('model.h5')
	
###print the keys contained in the history object
from keras.models import Model
import matplotlib.pyplot as plt
print(history_object.history.keys())
##Plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
