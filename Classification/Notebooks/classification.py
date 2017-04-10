from __future__ import print_function
import argparse
import numpy as np
from matplotlib import pyplot as plt

from scipy import misc

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.callbacks import Callback
from keras.backend import image_dim_ordering
from keras.utils.np_utils import to_categorical
import pandas as pd

import os

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--batchsize', type=int, default=10,
                    help='Batch Size')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of Epochs')
args = parser.parse_args()


shp = (len(os.listdir('../../data/Images/'))-1, ) + misc.imread('../../data/Images/1.png').shape
shp = shp[:3] + (3,)
img = np.empty(shp)

print(shp)

for i in range(len(os.listdir('../../data/Images/'))-1):
    img[i, :, :] = misc.imread('../../data/Images/' + str(i) +'.png', mode='RGB')
    
img /= 255.
labels = pd.read_table('../../data/Labels/labels.txt', header=None).values.flatten()
x_train = img[:75]
x_test = img[75:]

# input image dimensions
img_rows, img_cols = shp[1], shp[2]
# The CIFAR10 images are RGB.
img_channels = shp[3]
num_classes = 5


y_train = to_categorical(labels[:75]-1, num_classes)
y_test = to_categorical(labels[75:]-1, num_classes)
try_x = x_train.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2])
batch_size = args.batchsize
epochs = args.epochs

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        score = self.model.evaluate(x_train, y_train, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        
model = Sequential()
model.add(Conv2D(100, 3, 3,
                 activation='relu', init='glorot_normal',
                 input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, init='glorot_normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs,
          verbose=True, callbacks=[LossHistory()])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


