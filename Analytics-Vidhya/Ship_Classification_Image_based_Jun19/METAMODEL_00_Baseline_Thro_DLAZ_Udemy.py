# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:51:01 2019

@author: kzt9qh
"""
# Part - 1 Building a CNN

# importing keras libraries and packages
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import Callback
import tensorflow as tf
from keras.regularizers import l2

# Defining CallBack
DESIRED_ACCURACY = 0.99

class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=DESIRED_ACCURACY):
      print("\nReached "+str(DESIRED_ACCURACY)+"% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# initialising the CNN
classifier = Sequential()

# Step -1 Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(128,128,3), activation='relu',
                             kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu',
                             kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a convolutional layer
classifier.add(Convolution2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a convolutional layer
classifier.add(Convolution2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a convolutional layer
classifier.add(Convolution2D(512, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 2048, activation = 'relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
classifier.add(layers.Dropout(0.3))
classifier.add(Dense(units = 1024, activation = 'relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
classifier.add(layers.Dropout(0.3))
classifier.add(Dense(units = 512, activation = 'relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
classifier.add(layers.Dropout(0.3))
classifier.add(Dense(units = 256, activation = 'relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
classifier.add(layers.Dropout(0.3))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dense(units = 5, activation = tf.nn.softmax))

classifier.summary()

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.4,
                                   rotation_range=20,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('V:/2019/WIP/AV_GamesOfDL_Image_Classification/Images_Train',
                                                 target_size = (128, 128),
                                                 batch_size = 128,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('V:/2019/WIP/AV_GamesOfDL_Image_Classification/Images_Validation',
                                            target_size = (128, 128),
                                            batch_size = 75,
                                            class_mode = 'categorical')

history = classifier.fit_generator(training_set,
                         steps_per_epoch = 48,
                         epochs = 600,
                         validation_data = test_set,
                         validation_steps = 1,callbacks=[callbacks])
# Part 3 - Making new predictions
import os
address_TrainTest = r'V:\2019\WIP\AV_GamesOfDL_Image_Classification'
address_testImage = r'V:\2019\WIP\AV_GamesOfDL_Image_Classification\Images_Test'
testCSVFilePath = os.path.join(address_TrainTest, "test.csv")
Test_files = []
if os.path.exists(testCSVFilePath):
    with open(testCSVFilePath) as csvFile:
        Test_files = csvFile.readlines()
Test_files = [item.split(",") for item in Test_files[1:]]
import numpy as np
from keras.preprocessing import image
pred_Values=[]
softmax_Values = []
for picNumber in range(len(Test_files)):
    picName = Test_files[picNumber][0][:-1]
    ImagePredPath = os.path.join(address_testImage, picName)
    test_image = image.load_img(ImagePredPath, target_size = (128, 128))
    test_image = image.img_to_array(test_image)3
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    softmax_Values.append(result)
    max_Value = np.amax(result[0])
    max_value_index = np.where(result[0] == np.amax(result[0]))
    if max_Value == 0.0:
        index = 1
        pred_Values.append(index)
        print('What')
    else:
        index = max_value_index[0][0]+1
        pred_Values.append(index)
#    print(max_Value,max_value_index)
myarray = np.asarray(pred_Values)
training_set.class_indices

# ploting accuracy history
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
#plt.figure()
#plt.show()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
#plt.figure()
#plt.show()