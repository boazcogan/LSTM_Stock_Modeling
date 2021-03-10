import torch
import numpy as np
from math import sqrt


def return_loss(inputs, target):
    volatility_scaling = 1
    sig_tgt = .15
    return torch.mean(np.sign(target) * inputs) * -1


def mse_loss(output, target):
    volatility_scaling = 1
    loss = torch.mean((output - (target / volatility_scaling))**2)
    return loss


def binary_loss(inputs, target):
    volatility_scaling = 1
    loss = torch.nn.BCELoss(inputs, target)
    return loss


def sharpe_loss(inputs, target):
    n_days = 252
    # R_it is the return given by the targets
    R_it = torch.sum(target ** 2) / len(inputs)

    loss = torch.mean(inputs) * sqrt(n_days)
    loss /= torch.sqrt(torch.abs(R_it - torch.mean(torch.pow(inputs, 2))))

    return loss * -1
