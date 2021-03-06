import torch

import torch
from torch import nn
from src.Handler import *
from torch.autograd import Variable


class WaveNet(torch.nn.Module):
    def __init__(self, input_shape, num_layers, output_shape, kernel_size, dilation_depth):
        super(WaveNet, self).__init__()
        self.num_layers = num_layers
        self.dilations = [2**i for i in range(dilation_depth)]
        self.Conv1 = nn.Conv1d(in_channels=input_shape, out_channels=input_shape, kernel_size=1)
        self._conv_layers = []
        for i2 in self.dilations:
            self._conv_layers.append(nn.Conv1d(in_channels=input_shape,
                                               out_channels=input_shape,
                                               kernel_size=kernel_size,
                                               dilation=i2,
                                               padding='same'))
        self.linear = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        outs = self.Conv1(x)
        for elem in self._conv_layers:
            outs = elem(outs)
        outs = self.linear(outs)
        return outs


class WaveNetHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate):
        super(WaveNetHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate)

    def create_model(self, input_shape, hidden_shape, output_shape, num_layers):
        self.model = WaveNet(input_shape, hidden_shape, output_shape, num_layers)

    def train(self, x, y):
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.FloatTensor(y))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            pred = self.model.forward(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            print('Epoch {}:\t train loss: {}'.format(epoch, loss.item()))
            loss.backward()
            optimizer.step()
