"""
reference: https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
"""

import torch
from src.Handler import *
import src.CustomLoss as CustomLoss


class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        dropout1 = self.dropout1(hidden)
        relu = self.tanh(dropout1)
        fc2 = self.fc2(relu)
        dropout2 = self.dropout2(fc2)
        out = self.sigmoid(dropout2)
        return out


class MLPHandler(Handler):

    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
        super(MLPHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable, alpha)

    def create_model(self, input_shape, hidden_shape, output_shape):
        self.model = MLP(input_shape, hidden_shape, output_shape)

    def train(self, x, y):
        if self.loss_method == 'MSE':
            criterion = CustomLoss.mse_loss
        elif self.loss_method == 'Returns':
            criterion = CustomLoss.return_loss
        elif self.loss_method == 'Sharpe':
            criterion = CustomLoss.sharpe_loss
        else:
            raise Exception("Invalid loss method")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.eval()
        self.model.train()
        avg_losses = []

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, x.shape[0], self.batch_size):
                features = torch.FloatTensor(x[i:i+self.batch_size])
                labels = torch.FloatTensor(y[i:i+self.batch_size])
                l1reg = torch.tensor(0)
                optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(features)
                # Compute Loss
                loss = criterion(y_pred.squeeze(), labels)
                if self.l1enable:
                    for param in self.model.parameters():
                        l1reg += torch.norm(param, 1).long()
                    loss += self.alpha * l1reg

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