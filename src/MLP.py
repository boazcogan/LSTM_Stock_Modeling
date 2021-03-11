"""
reference: https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
"""

import torch
from src.Handler import *
import src.CustomLoss as CustomLoss
from torch.autograd import Variable


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron. As described in the academic paper it has a single hidden layer utilizes a dropout of 0.5
    and uses the tanh activation function since we are looking at the direct outputs of the model.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.tanh1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.tanh2 = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        dropout1 = self.dropout1(hidden)
        tanh1 = self.tanh1(dropout1)
        fc2 = self.fc2(tanh1)
        dropout2 = self.dropout2(fc2)
        out = self.tanh2(dropout2)
        return out


class MLPHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
        super(MLPHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable, alpha)

    def create_model(self, input_shape, hidden_shape, output_shape, dropout):
        self.model = MLP(input_shape, hidden_shape, output_shape, dropout)
