import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

TEST_PATH = 'data/test.csv'
ID_NAME = 'key'
PREDICATION_NAME = 'fare_amount'
SCORE_PATH = 'score_master_file.txt'  ## the file which stores model name and score

def output_submission(prediction, submission_file):
    """
    generate the submission file given prediction

    Args:
        prediction: numpy array of predictions
        file_name: the location and name of the submission file
    
    Writes:
        submission file
    """
    raw_test = pd.read_csv(TEST_PATH, usecols = [ID_NAME])
    df = pd.DataFrame(prediction, columns=[PREDICATION_NAME])
    df[ID_NAME] = raw_test[ID_NAME]
    df[[ID_NAME, PREDICATION_NAME]].to_csv((submission_file), index=False)
    print('Output complete')
     
def plot_loss(history, file_name):
    """
    plot (MSE, RMSE) v.s. epoch output by keras

    Args:
        history object of keras (output of model.fit)
        file_name: location and name of the figure

    Saves:
        (MSE, RMSE) v.s. training epoch plot
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('model loss: MSE')
    axs[0].set_ylabel('MSE')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper right')

    axs[1].plot([sqrt(i) for i in history.history['loss']])
    axs[1].plot([sqrt(i) for i in history.history['val_loss']])
    axs[1].set_title('model metric: RMSE')
    axs[1].set_ylabel('RMSE')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper right')
    
    plt.savefig(file_name)

def write_score(model_name, history):
    """
    write model_name and model validation score to SCORE_PATH
    """
    with open (SCORE_PATH, 'a') as f:
        f.write('\n')
        f.write("{}: {}".format(model_name, sqrt(history.history['val_loss'][-1])))





