import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from data.GetDataset import normalized_returns
from math import sqrt

class MyCustomLoss(nn.Module):
    def __init__(self, alpha=2, beta=0.5, method='binary', regularization='none', model=None):
        assert method in ['binary', 'reg', 'sharpe', 'return'],\
            "The method must be one of ['binary', 'reg', 'sharpe', 'return']"
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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
        z = torch.div(torch.sum(torch.pow(inputs, 2)), len(target))
        return torch.div(torch.mul(ret_loss, sqrt(252)), torch.sqrt(z - torch.pow(ret_loss, 2)))

    def return_loss(self, inputs, target):
        return torch.div(torch.sum(torch.mul(torch.mul(np.sign(target), .15) / 1, (target[len(target)-1][0] - target[0][0]))), len(target))

    def get_regularization(self, loss):
        return self.alpha * torch.norm(self.model, 1) + loss if self.regularization == 'L1' else loss
