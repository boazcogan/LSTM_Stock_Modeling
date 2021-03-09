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
        self.relu = torch.nn.ReLU()
        self.linear = nn.Linear(hidden_shape, output_shape)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, h_n, c_n):
        output, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        output = output.view(-1, self.hidden_shape)
        dropout1 = self.dropout1(output)
        activ1 = self.relu(dropout1)
        pred = self.linear(activ1)
        dropout2 = self.dropout2(pred)
        activ2 = self.tanh(dropout2)
        return activ2, h_n, c_n


class LSTMHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False):
        super(LSTMHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable)

    def create_model(self, input_shape, hidden_shape, output_shape, num_layers):
        self.model = LSTM(input_shape, hidden_shape, output_shape, num_layers)

    def train(self, x, y):
        avg_losses = []
        criterion = sharpe_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(x.shape[0]):
                features = Variable(torch.FloatTensor(x[i].astype(np.float32)))
                labels = Variable(torch.FloatTensor(y[i].astype(np.float32)))
                features = torch.reshape(features, (features.shape[0], 1, features.shape[1]))
                h_n = Variable(torch.zeros(self.model.num_layers, self.batch_size, self.model.hidden_shape))
                c_n = Variable(torch.zeros(self.model.num_layers, self.batch_size, self.model.hidden_shape))

                for j in range(0, x[i].shape[0]-x[i].shape[0]%self.batch_size, self.batch_size):
                    features = Variable(torch.FloatTensor(x[i][j:j+self.batch_size].astype(np.float32)))
                    labels = Variable(torch.FloatTensor(y[i][j:j+self.batch_size].astype(np.float32)))
                    features = torch.reshape(features, (features.shape[0], 1, features.shape[1]))

                    pred, h_n, c_n = self.model.forward(features, h_n, c_n)
                    h_n = h_n.detach()
                    c_n = c_n.detach()
                    l1reg = torch.tensor(0)
                    optimizer.zero_grad()
                    loss = criterion(pred, labels)
                    if self.l1enable:
                        for param in self.model.parameters():
                            l1reg += torch.norm(param, 1).long()
                        loss += l1reg
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach().numpy()
            print('Epoch {}:\t train loss: {}'.format(epoch, total_loss / x.shape[0]))
            avg_losses.append(total_loss / x.shape[0])
        return avg_losses

    def predict(self, data):
        x = Variable(torch.FloatTensor(data))
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        pred = self.model.forward(x)
        return pred

    def test(self, x, y):
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.FloatTensor(y))
        h_0 = Variable(torch.zeros(self.model.num_layers, x.size(0), self.model.hidden_shape))
        c_0 = Variable(torch.zeros(self.model.num_layers, x.size(0), self.model.hidden_shape))

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        criterion = torch.nn.MSELoss()
        pred, _, _ = self.model.forward(x, h_0, c_0)
        loss = criterion(pred, y)
        return loss, pred

def return_loss(inputs, target):
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
