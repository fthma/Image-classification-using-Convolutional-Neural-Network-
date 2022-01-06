import keras
import keras.utils
import sklearn
from sklearn.model_selection import cross_validate
import sklearn.metrics
import matplotlib.pyplot as plt
from keras.utils import to_categorical
#from keras import utils as np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, MaxPool2D, Flatten,MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
#import tensorflow as tf
import numpy as np
import cv2
import os

# Importing the required Keras modules containing model and layers

from keras.layers import   MaxPooling2D

#importing the mnist dataset

import tensorflow as tf
(MNIST_TrainData, MNIST_TrainLabel), (MNIST_TestData, MNIST_TestLabel) = tf.keras.datasets.mnist.load_data()

def trainMNISTData(x_train, y_train,x_test, y_test):
    
    # Reshaping the array 
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalizing the images
    x_train /= 255
    x_test /= 255
    
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size=(3,3),padding = 'Same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
      
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.2))
      
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.2))
      
    model.add(Dense(10, activation = "softmax"))
    
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=10)
    
    model.evaluate(x_test, y_test)
    
print("Training the MNIST dataset")    
trainMNISTData(MNIST_TrainData, MNIST_TrainLabel,MNIST_TestData, MNIST_TestLabel) 

#load the 15-scene dataset
DATADIR='15-Scene'
Categories=os.listdir(DATADIR)
dataset_x = []
dataset_y = []
LabelsArray = np.eye(15)

for category in Categories:
    path=os.path.join(DATADIR,category)
    for fname in os.listdir(path):
        img = cv2.imread(path+'/'+fname, 2)
        img = cv2.resize(img, (32,32))
        dataset_x.append(np.reshape(img, [32,32,1]))
        dataset_y.append(np.reshape(LabelsArray[int(category)], [15]))
dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)


"""shuffle dataset"""
p = np.random.permutation(len(dataset_x))
dataset_x = dataset_x[p]
dataset_y = dataset_y[p]


X_test = dataset_x[:int(len(dataset_x)*0.3)]
Y_test = dataset_y[:int(len(dataset_x)*0.3)]
X_train = dataset_x[int(len(dataset_x)*0.3):]
Y_train = dataset_y[int(len(dataset_x)*0.3):]

def CNN15Scene(a,b,c,d):
    batch_size = 128
    num_classes = 15
    epochs = 100
    img_rows, img_cols = X_train.shape[1],X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu', input_shape = input_shape))
    
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation = "softmax"))
    
    
    
    # Define the optimizer
    #optimizer = Adam(lr=0.0005, decay=1e-6)
    model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["acc"])
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(a)
    history = model.fit_generator(datagen.flow(a,b, batch_size=64),steps_per_epoch=len(a) / 64, epochs=epochs)
    score = model.evaluate(c,d, verbose=1)
    print('\nAccuracy:', score[1])
    
print("---------------------")
print("15 Scene dataset")
CNN15Scene(X_train, Y_train, X_test, Y_test)