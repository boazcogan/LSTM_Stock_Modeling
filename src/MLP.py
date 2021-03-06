"""
reference: https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
"""

import torch
from src.Handler import *
import numpy as np
from math import sqrt


class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.tanh(hidden)
        out = self.fc2(relu)
        out = self.sigmoid(out)
        return out


class MLPHandler(Handler):

    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size):
        super(MLPHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size)

    def create_model(self, input_shape, hidden_shape, output_shape):
        self.model = MLP(input_shape, hidden_shape, output_shape)

    def train(self, x, y):
        criterion = mse_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.eval()
        # y_pred = self.model(x)
        # before_train = criterion(y_pred.squeeze(), y)
        # print('Test loss before training', before_train.item())
        self.model.train()
        avg_losses = []

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, x.shape[0], self.batch_size):
                features = torch.FloatTensor(x[i:i+self.batch_size])
                labels = torch.FloatTensor(y[i:i+self.batch_size])
                optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(features)
                # Compute Loss
                loss = criterion(y_pred.squeeze(), labels)

                # print('Epoch {}:\t train loss: {}'.format(epoch, loss.item()))
                # avg_losses.append(loss.item())
                # Backward pass
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_losses.append(total_loss / x.size)
            print('epoch {}:\t loss {}'.format(epoch, total_loss / x.size))
        return avg_losses

    def test(self, x, y):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        criterion = torch.nn.MSELoss()
        y_pred = self.model(x)
        loss = criterion(y_pred.squeeze(), y)
        return loss, y_pred

def return_loss(inputs, target):
    # page 3, equation 1: sig_t^i is the ex-ante volatility estimate
    # not sure how to implement in our context; dividing by 1 where sig_t^i should be'
    volatility_scaling = 1
    sig_tgt = .15
    return torch.mean(np.sign(target) * inputs) * -1

def mse_loss(output, target):
    volatility_scaling = 1
    loss = torch.mean((output - (target / volatility_scaling))**2)
    return loss

def binary_loss(inputs, target):
    volatility_scaling = 1
    loss = torch.log(target)

def sharpe_loss(inputs, target):
    n_days = 252
    # R_it is the return given by the targets
    R_it = torch.sum(target ** 2) / len(inputs)

    loss = torch.mean(inputs) * sqrt(n_days)
    loss /= torch.sqrt(torch.abs(R_it - torch.mean(torch.pow(inputs, 2))))

    return loss * -1