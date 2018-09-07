"""
this handles all models, simple ones, and ensemble ones
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from input import input
from utils import *

class DenseModel():
    def __init__(self, feature_dim):
        """
        init the dense model, define training parameters and build model
        """
        # Training parameters
        self.batch_size = 512
        self.epochs = 50
        self.learning_rate = 0.001
        self.feature_dim = feature_dim

        # model
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(256, activation='relu', input_dim=self.feature_dim))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(1024, activation='relu'))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dropout(0.2))
        # model.add(keras.layers.Dense(2048, activation='relu'))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(256, activation='relu'))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(1))

        adam = tf.train.AdamOptimizer(self.learning_rate)
        self.model.compile(loss='mse', optimizer=adam, metrics=['mae'])
        # early = EarlyStopping(monitor='val_loss', patience=15, mode='min')

    def train(self, X_train, y_train, X_valid, y_valid, feature_names):
        """
        train the model, return training history
        """
        self.training_size = X_train.shape[0]
        self.feature_names = feature_names
        self.history = self.model.fit(x=X_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, 
                    validation_data=(X_valid, y_valid), shuffle=True)
        return self.history

    def predict(self, X_test):
        """
        predict of test data
        """
        prediction = self.model.predict(X_test, batch_size=self.batch_size, verbose=1)
        return prediction

    def print_summary(self, filename):
        """
        print model information to file
        """
        with open (filename, 'w') as f:
            f.write('Training parameters: \n')
            f.write('Epochs: {}\n'.format(self.epochs))
            f.write('Learning rate: {}\n'.format(self.learning_rate))
            f.write('Batch size: {}\n'.format(self.batch_size))
            f.write('\n')
            
            f.write('Data info: \n')
            f.write('Dataset size: {}\n'.format(self.training_size))
            f.write('Input dimension: {}\n'.format(self.feature_dim))
            f.write('Features used: {}\n'.format(self.feature_names))
            f.write('\n')

            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
if __name__=='__main__':
    model_name = 'dense'+'_'+'test'
    submission_file = 'submission/submission_'+model_name+'.csv'
    figure_name = 'figures/loss_metric_'+model_name+'.png'
    summary_file = 'model_summary/summary_'+model_name+'.txt'

    X_train, y_train, X_test, feature_names = input()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

    model = DenseModel(feature_dim=X_train.shape[1])
    history = model.train(X_train, y_train, X_valid, y_valid, feature_names)
    prediction = model.predict(X_test)

    model.print_summary(summary_file)
    plot_loss(history, figure_name)
    output_submission(prediction, submission_file)
    write_score(model_name, history)








