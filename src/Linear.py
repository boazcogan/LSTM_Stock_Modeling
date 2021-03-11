"""
referenced source: https://d2l.ai/
"""
import torch
from torch.autograd import Variable
import src.CustomLoss as CustomLoss
from src.Handler import *


class Linear(torch.nn.Module):
    """
    The simplest example, a linear classifier.
    """

    def __init__(self, input_size, output_size, dropout):
        """
        Default constructor for the Linear classifier
        :param input_size: the input shape to instantiate the model with
        :param the output shape for the model
        :param epochs: the number of iterations to pass over the training data
        """
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = self.linear(x)
        dropout = self.dropout(out)
        activ = self.tanh(dropout)
        return activ


class LinearHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
        super(LinearHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable, alpha)

    def create_model(self, input_shape, output_shape, dropout):
        self.model = Linear(input_shape, output_shape, dropout)
