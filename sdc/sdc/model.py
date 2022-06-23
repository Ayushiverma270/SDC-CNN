import scipy.io as io
import h5py
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Conv2D,Dropout,Cropping2D,Convolution2D ,BatchNormalization,MaxPooling2D
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from keras.optimizers import Adam
import numpy as np
import cv2

def load_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255 -0.5, input_shape=(240,320,3)))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(BatchNormalization())
    #model.add(Dropout(1))
    #model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    #model.add(BatchNormalization())
    #model.add(Dropout(1))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2)))
    #model.add(Conv2D(64, 3, 3, activation='elu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(1))
    model.add(Flatten())
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(1))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    #model.summary() #prints the architecture of the model
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model
