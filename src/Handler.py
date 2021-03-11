import torch
import src.CustomLoss as CustomLoss
from torch.autograd import Variable


class Handler:
    """
    A simple handler to manage the hyperparameters and training loops of the neural network classes.
    """
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
        self.epochs = epochs
        self.loss_method = loss_method
        self.regularization_method = regularization_method
        self.learning_rate = learning_rate
        self.model = None
        self.batch_size = batch_size
        self.l1enable = l1enable
        self.alpha = alpha

    def train(self, x, y):
        """
        basic training loop for the networks, pulls the loss function from the custom loss code block
        :param x: the features
        :param y: the labels
        :return: average losses
        """
        if self.loss_method == 'MSE':
            criterion = CustomLoss.mse_loss
        elif self.loss_method == 'Returns':
            criterion = CustomLoss.return_loss
        elif self.loss_method == 'Sharpe':
            criterion = CustomLoss.sharpe_loss
        else:
            raise Exception("Invalid loss method")
        # The optimizer specified in the academic paper
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        avg_losses = []
        # for each epoch
        for epoch in range(self.epochs):
            total_loss = 0
            # for each batch
            for i in range(0, x.shape[0], self.batch_size):
                # extract the batch and train on it
                inputs = Variable(torch.from_numpy(x[i:i+self.batch_size]))
                labels = Variable(torch.from_numpy(y[i:i+self.batch_size]))
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # if l1enable is defined then apply l1regularization
                if self.l1enable:
                    l1reg = torch.tensor(0)
                    for param in self.model.parameters():
                        l1reg += torch.norm(param, 1).long()
                    loss += self.alpha * l1reg
                # compute the backward propogation step and zero out the gradient so that we can train on the next batch
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            avg_losses.append(total_loss / x.size)
            print('epoch {}:\t loss {}'.format(epoch, total_loss / x.size))
        return avg_losses

    def test(self, x, y):
        """
        Test the model to evaluate performance
        :param x: features
        :param y: targets
        :return: loss, predictions
        """
        # same as training but without the back-prop step and no batches.
        if self.loss_method == "MSE":
            criterion = CustomLoss.mse_loss
        elif self.loss_method == "Custom_Sharpe":
            criterion = CustomLoss.sharpe_loss
        elif self.loss_method == "Returns":
            criterion = CustomLoss.return_loss
        else:
            print("Loss method not recognized, defaulting to MSE")
            criterion = torch.nn.MSELoss()
        inputs = Variable(torch.from_numpy(x))
        labels = Variable(torch.from_numpy(y))
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        return loss, outputs
