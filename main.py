import data.GetDataset as gd
from src import *
import torch
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler


if __name__ == '__main__':
    """
    train_features, train_targets, test_features, test_targets = gd.get_aggregated_dataset("commodities", 5, 7, 0.9)
    batch_size = 32
    print('----- Linear -----')
    linear = LinearHandler(100, "MSE", None, 0.01, batch_size)
    linear.create_model(train_features.shape[1], 1)
    linear_losses = linear.train(train_features, train_targets)
    plt.figure(0)
    plt.title("Linear Loss")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.xlabel("epoch")
    plt.plot(linear_losses)
    plt.show()
    linear.test(test_features, test_targets)
    # print(f"\nThe MSE for the Linear test set is: {linear.test(test_features, test_targets)}\n\n")

    print('\n\n\n----- MLP -----')
    mlp = MLPHandler(100, "MSE", None, 0.01, batch_size)
    # they never specify the number of hidden nodes
    # page 6: "a 2-layer neural network can be used to incorporate non-linear effects."
    # Looks like just 1 hidden + 1 output
    mlp.create_model(train_features.shape[1], 15, 1)
    mlp_losses = mlp.train(train_features, np.squeeze(train_targets))
    plt.figure(0)
    plt.title("MLP Loss")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.xlabel("epoch")
    plt.plot(mlp_losses)
    plt.show()
    mlp.test(test_features, test_targets)
    # print(f"\nThe MSE for the MLP test set is: {linear.test(test_features, np.squeeze(test_targets))}\n\n")
    """
    train_features, train_targets, test_features, test_targets = gd.get_normal_dataset("commodities", 0.9, 7, normalize=True)

    print('\n\n\n----- LSTM -----')
    lstm = LSTMHandler(100, "MSE", None, 0.01, 0)
    lstm.create_model(train_features.shape[1], 15, 1, 1)
    lstm_losses = lstm.train(train_features, train_targets)
    plt.figure(0)
    plt.title("lstm Loss")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.xlabel("epoch")
    plt.plot(lstm_losses)
    plt.show()
    # print(f"\nThe MSE for the LSTM test set is: {linear.test(test_features, test_targets)}\n\n")
