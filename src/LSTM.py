"""
reference: https://cnvrg.io/pytorch-lstm/
"""
import torch
from torch import nn
from src.Handler import *
from torch.autograd import Variable
import numpy as np
from math import sqrt
from src.CustomLoss import MyCustomLoss


class LSTM(torch.nn.Module):
    def __init__(self, input_shape, hidden_shape, num_layers, output_shape):
        super(LSTM, self).__init__()
        self.hidden_shape = hidden_shape
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=hidden_shape,num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_shape, output_shape)
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_shape),
        #                     torch.zeros(1, 1, self.hidden_shape))

    def forward(self, x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_shape))
        # current state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_shape))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_shape)
        pred = self.linear(hn)
        return pred


class LSTMHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size):
        super(LSTMHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size)

    def create_model(self, input_shape, hidden_shape, output_shape, num_layers):
        self.model = LSTM(input_shape, hidden_shape, output_shape, num_layers)

    def train(self, x, y):
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.FloatTensor(y))

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        avg_losses = []
        criterion = mse_loss
        # criterion = MyCustomLoss(method=self.loss_method)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            pred = self.model.forward(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            print('Epoch {}:\t train loss: {}'.format(epoch, loss))
            avg_losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
        return avg_losses

    def predict(self, data):
        x = Variable(torch.FloatTensor(data))
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        pred = self.model.forward(x)
        return pred

    def test(self, x, y):
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.FloatTensor(y))

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        criterion = torch.nn.MSELoss()
        pred = self.model.forward(x)
        loss = criterion(pred, y)
        return loss, pred

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