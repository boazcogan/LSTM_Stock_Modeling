"""
reference: https://cnvrg.io/pytorch-lstm/
"""
import torch
from torch import nn
from src.Handler import *
from torch.autograd import Variable


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
    def __init__(self, epochs, loss_method, regularization_method, learning_rate):
        super(LSTMHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate)

    def create_model(self, input_shape, hidden_shape, output_shape, num_layers):
        self.model = LSTM(input_shape, hidden_shape, output_shape, num_layers)

    def train(self, x, y):
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.FloatTensor(y))

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            pred = self.model.forward(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            print('Epoch {}:\t train loss: {}'.format(epoch, loss.item()))
            loss.backward()
            optimizer.step()
