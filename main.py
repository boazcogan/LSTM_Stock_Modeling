import data.GetDataset as gd
from src import *
import torch
import numpy as np

if __name__ == '__main__':
    train_features, train_targets, test_features, test_targets = gd.get_dataset("fx", 5, 7, 0.9)
    linear = LinearHandler(100, "MSE", None, 0.01)
    linear.create_model(train_features.shape[1], 1)
    linear.train(train_features, train_targets)

    mlp = MLPHandler(100, "MSE", None, 0.01)
    # they never specify the number of hidden nodes
    mlp.create_model(train_features.shape[1], 15, 1)
    mlp.train(train_features, np.squeeze(train_targets))

    lstm = LSTMHandler(100, "MSE", None, 0.01)
    lstm.create_model(train_features.shape[1], 15, 1, 1)
    lstm.train(train_features, train_targets)
