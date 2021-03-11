"""
reference: https://cnvrg.io/pytorch-lstm/
"""
import torch
from torch import nn
from src.Handler import *
from torch.autograd import Variable
import numpy as np
import src.CustomLoss as CustomLoss


class LSTM(torch.nn.Module):
    """
    The LSTM model as defined by the academic paper. There is a single hidden unit within the LSTM, tanh activation
    functions, and a single fully connected layer.
    """
    def __init__(self, input_shape, hidden_shape, num_layers, output_shape, dropout):
        super(LSTM, self).__init__()
        self.hidden_shape = hidden_shape
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=hidden_shape,num_layers=num_layers, batch_first=True)
        self.tanh1 = torch.nn.Tanh()
        self.linear = nn.Linear(hidden_shape, output_shape)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.tanh2 = torch.nn.Tanh()

    def forward(self, x, h_n, c_n):
        """
        Forward pass of the LSTM, takes the hidden state and the cell state as inputs
        :param x: features
        :param h_n: hidden state
        :param c_n: cell state
        :return: predictions of the lstm, hidden state, and the cell state
        """
        output, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        output = output.view(-1, self.hidden_shape)
        dropout1 = self.dropout1(output)
        activ1 = self.tanh1(dropout1)
        pred = self.linear(activ1)
        dropout2 = self.dropout2(pred)
        activ2 = self.tanh2(dropout2)
        return activ2, h_n, c_n


class LSTMHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
        super(LSTMHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable, alpha)

    def create_model(self, input_shape, hidden_shape, output_shape, num_layers, dropout):
        self.model = LSTM(input_shape, hidden_shape, output_shape, num_layers, dropout)

    def train(self, x, y):
        """
        Training loop for the LSTM
        :param x: features
        :param y: labels
        :return: The loss and predictions of the LSTM
        """
        avg_losses = []
        if self.loss_method == 'MSE':
            criterion = CustomLoss.mse_loss
        elif self.loss_method == 'Returns':
            criterion = CustomLoss.return_loss
        elif self.loss_method == 'Sharpe':
            criterion = CustomLoss.sharpe_loss
        else:
            raise Exception("Invalid loss method")
        # using the ADAM optimizer as specified by the paper
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # for each epoch
        for epoch in range(self.epochs):
            total_loss = 0
            # for each different sequence to train on
            for i in range(len(x)):
                # create a hidden state and cell state, instantiate them to 0 to symbolize the start of a sequence
                h_n = Variable(torch.zeros(self.model.num_layers, self.batch_size, self.model.hidden_shape))
                c_n = Variable(torch.zeros(self.model.num_layers, self.batch_size, self.model.hidden_shape))

                # For each batch
                for j in range(0, x[i].shape[0]-x[i].shape[0] % self.batch_size, self.batch_size):
                    # get the features and targets to train on
                    features = Variable(torch.FloatTensor(x[i][j:j+self.batch_size].astype(np.float32)))
                    labels = Variable(torch.FloatTensor(y[i][j:j+self.batch_size].astype(np.float32)))
                    # its organized as batch first so reshape
                    features = torch.reshape(features, (features.shape[0], 1, features.shape[1]))
                    # get the predictions and hidden/cell states
                    pred, h_n, c_n = self.model.forward(features, h_n, c_n)
                    # detach them so that we can feed them back in to the model on the next iteration
                    h_n = h_n.detach()
                    c_n = c_n.detach()
                    loss = criterion(pred, labels)
                    if self.l1enable:
                        l1reg = torch.tensor(0)
                        for param in self.model.parameters():
                            l1reg += torch.norm(param, 1).long()
                        loss += self.alpha * l1reg
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.detach().numpy()
            print('Epoch {}:\t train loss: {}'.format(epoch, total_loss / len(x)))
            avg_losses.append(total_loss / len(x))
        return avg_losses

    def test(self, x, y):
        """
        Test the network on a single sequence only
        :param x: the features of a single sequence.
        :param y: the targets for a single sequence.
        :return: the loss and predictions for the sequence.
        """
        # for the testing loop maintain the hidden and cell state and iterate over every training sample
        h_n = Variable(torch.zeros(self.model.num_layers, 1, self.model.hidden_shape))
        c_n = Variable(torch.zeros(self.model.num_layers, 1, self.model.hidden_shape))
        total_loss = 0
        preds = []
        for i in range(len(x)):
            features = Variable(torch.FloatTensor([x[i]]))
            targets = Variable(torch.FloatTensor([y[i]]))
            features = torch.reshape(features, (features.shape[0], 1, features.shape[1]))
            criterion = torch.nn.MSELoss()
            pred, h_n, c_n = self.model.forward(features, h_n, c_n)
            preds.append(pred.detach().numpy())
            h_n = h_n.detach()
            c_n = c_n.detach()
            total_loss += criterion(pred, targets).detach().numpy()
        loss = total_loss/len(x)
        return loss, np.array(preds)

