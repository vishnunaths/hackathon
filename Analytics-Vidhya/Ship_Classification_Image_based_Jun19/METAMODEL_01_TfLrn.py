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
from keras import Model
import tensorflow as tf

# Defining CallBack
DESIRED_ACCURACY = 0.99

class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=DESIRED_ACCURACY):
      print("\nReached "+str(DESIRED_ACCURACY)+"% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# Transfer Learning
import os
from keras.applications.inception_v3 import InceptionV3

local_weights_file = r'V:\2019\WIP\AV_GamesOfDL_Image_Classification\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (128, 128, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
#pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
classifier = layers.Flatten()(last_output)

# Full connection
classifier = layers.Dense(1024, activation='relu')(classifier)
classifier = layers.Dropout(0.2)(classifier)   
classifier = layers.Dense(512, activation='relu')(classifier)
classifier = layers.Dropout(0.2)(classifier)
#classifier = layers.Dense(256, activation='relu')(classifier)
#classifier = layers.Dropout(0.2)(classifier)
classifier = layers.Dense(128, activation='relu')(classifier)
#classifier = layers.Dense(64, activation='relu')(classifier)
#classifier = layers.Dense(32, activation='relu')(classifier)
#classifier = layers.Dense(16, activation='relu')(classifier)       
classifier = layers.Dense(5, activation=tf.nn.softmax)(classifier)

model = Model(pre_trained_model.input, classifier) 

# Compiling the CNN
from keras import optimizers
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   rotation_range=20,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('V:/2019/WIP/AV_GamesOfDL_Image_Classification/Images_Train',
                                                 target_size = (128, 128),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('V:/2019/WIP/AV_GamesOfDL_Image_Classification/Images_Validation',
                                            target_size = (128, 128),
                                            batch_size = 50,
                                            class_mode = 'categorical')

history = model.fit_generator(training_set,
                         steps_per_epoch = 98,
                         epochs = 500,
                         validation_data = test_set,
                         validation_steps = 1,callbacks=[callbacks])

# Part 3 - Making new predictions
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
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
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
plt.figure()
plt.show()

# Directions
from keras.applications.vgg16 import vgg16 
keras.applications.inception_v3
#Nadam