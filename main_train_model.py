"""
    train the GMM_HMM and XGB_HMM model
    then generate the state_proba
    then train the LSTM model
"""

from train_model.train_HMM_model import train_HMM_model
from train_model.train_LSTM_model import train_LSTM_model

if __name__ == '__main__':

    n_states = 3
    train_HMM_model(n_states)

    train_LSTM_model()
