from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import regularizers
import tensorflow as tf

from input import input
from utils import *

# Training parameters
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.001
NROWS = 100000
NFEATURE = 20

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(64, activation='relu', input_dim=NFEATURE))
# Add another:
model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

import numpy as np

data = np.random.random((NROWS, NFEATURE))
labels = np.random.random((NROWS, 10))

model.fit(data, labels, epochs=10, batch_size=32)