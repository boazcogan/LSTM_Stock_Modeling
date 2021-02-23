import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyCustomLoss(nn.Module):
    def __init__(self, alpha=2, method='binary', regularization='none', model=None):
        assert method in ['binary', 'reg', 'sharpe', 'return'],\
            "The method must be one of ['binary', 'reg', 'sharpe', 'return']"
        super().__init__()
        self.alpha = alpha
        if method == "binary":
            self.loss_func = self.binary_loss()
        elif method == "reg":
            self.loss_func = self.reg_loss()
        elif method == "sharpe":
            self.loss_func = self.sharpe_loss()
        else:
            self.loss_func = self.return_loss()

        self.regularization = regularization
        self.model = model

    def forward(self, inputs, target):
        """
        implement whatever loss function is used here
        :param inputs:
        :param target:
        :return:
        """
        loss = self.get_regularization(self.loss_func(inputs, target))
        return loss

    @staticmethod
    def binary_loss():
        return torch.nn.BCELoss()


    @staticmethod
    def reg_loss():
        return torch.nn.MSELoss()

    def sharpe_loss(self):
        return self.binary_loss()

    def return_loss(self):
        return self.binary_loss()

    def get_regularization(self, loss):
        if self.regularization == 'L1':
            return self.alpha * torch.norm(self.model, 1) + loss
