"""

referenced source: https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
"""
import torch
from torch.autograd import Variable
from src.CustomLoss import *
from src.Handler import *

class Linear(torch.nn.Module):
    """
    The simplest example, a linear classifier.
    """

    def __init__(self, input_size, output_size):
        """
        Default consructor for the Linear classifier
        :param input_size: the input shape to intantiate the model with
        :param the output shape for the model
        :param epochs: the number of iterations to pass over the training data
        """
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


class LinearHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate):
        super(LinearHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate)

    def create_model(self, input_shape, output_shape):
        self.model = Linear(input_shape, output_shape)

    def train(self, x, y):
        if self.loss_method == "MSE":
            criterion = torch.nn.MSELoss()
        else:
            print("Loss method not recognized, defaulting to MSE")
            criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        inputs = Variable(torch.from_numpy(x))
        labels = Variable(torch.from_numpy(y))
        for epoch in range(self.epochs):
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))
