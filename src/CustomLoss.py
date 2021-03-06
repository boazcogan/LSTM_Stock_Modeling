import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from data.GetDataset import normalized_returns
from math import sqrt

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
            self.loss_func = self.sharpe_loss
        else:
            self.loss_func = self.return_loss

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

    def sharpe_loss(self, inputs, target):
        ret_loss = self.return_loss(inputs, target)
        # same 1 placeholder as in self.return_loss()
        z = torch.sum(np.sign(target) * .15 / 1 * (target[len(target)-1][0] - target[0][0]) ** 2) / len(target)
        #print(z)
        #print(ret_loss * sqrt(252) / sqrt(z - ret_loss ** 2))
        return ret_loss * sqrt(252) / sqrt(z - ret_loss ** 2)

    def return_loss(self, inputs, target):
        # page 3, equation 1: sig_t^i is the ex-ante volatility estimate
        # not sure how to implement in our context; dividing by 1 where sig_t^i should be

        return torch.sum(np.sign(target) * .15 / 1 * (target[len(target)-1][0] - target[0][0])) / len(target)

    def get_regularization(self, loss):
        if self.regularization == 'L1':
            return self.alpha * torch.norm(self.model, 1) + loss