"""
this handles all models, simple ones, and ensemble ones
"""

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import regularizers

from input import input
from utils import *

# Training parameters
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.001

def dense_model(X_train, y_train, X_valid, y_valid, X_test):
	#model
	model = Sequential()
	model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
	model.add(BatchNormalization())
	#model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.2))
	model.add(Dense(32, activation='relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.2))
	model.add(Dense(16, activation='relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.2))
	model.add(Dense(1))

	adam = optimizers.adam(lr=LEARNING_RATE)
	model.compile(loss='mse', optimizer=adam, metrics=['mae'])
	# early = EarlyStopping(monitor='val_loss', patience=15, mode='min')

	#train
	history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, 
                    validation_data=(X_valid, y_valid), shuffle=True)

	#predict
	prediction = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
	return history, prediction

if __name__=='__main__':
	model_name = 'dense'
	submission_file = 'submission/submission_'+model_name+'.csv'
	figure_name = 'figures/loss_metric_'+model_name+'.jpg'

	X_train, y_train, X_test = input(1000000)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

	history, prediction = dense_model(X_train, y_train, X_valid, y_valid, X_test)
	plot_loss(history, figure_name)
	output_submission(prediction, submission_file)
	write_score(model_name, history)








