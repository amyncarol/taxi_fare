from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from input import input
from utils import *

# Training parameters
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
NROWS = 1000000
NFEATURE = 20

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(64, activation='relu', input_dim=NFEATURE))
# Add another:
model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

import numpy as np

data = np.random.random((NROWS, NFEATURE))
labels = np.random.random((NROWS, 10))

model.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# # # Instantiates a toy dataset instance:
# dataset = tf.data.Dataset.from_tensor_slices((data, labels))
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.repeat()

# # # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
# model.fit(dataset, epochs=EPOCHS, steps_per_epoch=NROWS//BATCH_SIZE)