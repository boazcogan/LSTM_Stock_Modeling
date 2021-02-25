import data.GetDataset as gd
from src import *
import torch
import numpy as np

if __name__ == '__main__':
    print('----- Linear -----')
    train_features, train_targets, test_features, test_targets = gd.get_dataset("fx", 5, 7, 0.9)
    linear = LinearHandler(100, "MSE", None, 0.01)
    linear.create_model(train_features.shape[1], 1)
    linear.train(train_features, train_targets)

    print('----- MLP -----')
    mlp = MLPHandler(100, "MSE", None, 0.01)
    # they never specify the number of hidden nodes
    # page 6: "a 2-layer neural network can be used to incorporate non-linear effects." Looks like just 1 hidden + 1 output
    mlp.create_model(train_features.shape[1], 15, 1)
    mlp.train(train_features, np.squeeze(train_targets))

    print('----- LSTM -----')
    lstm = LSTMHandler(100, "MSE", None, 0.01)
    lstm.create_model(train_features.shape[1], 15, 1, 1)
    lstm.train(train_features, train_targets)
