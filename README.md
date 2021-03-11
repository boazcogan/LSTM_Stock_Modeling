## Replicating *Enhancing Time Series Momentum Strategies Using Deep Neural Networks* - Project for CS496: Advanced Topics in Deep Learning

### Sections
1. Introduction
2. CLC Dataset
3.

### Introduction
Time series momentum, or trend momentum, is measured by a portfolio comprised of long assets that have had recent positive returns and short assets that have had recent negative returns. Academic evidence suggests that strategies that optimize time series momentum improve a portfolio's risk-adjusted returns.

These time series momentum strategies have two main attributes, trend estimation and position sizing; and recent studies have implemented various supervised learning models to predict these components. These models, however, do not account for volatility and other risk characteristics, and require manual specifications.

In *Enhancing Time Series Momentum Strategies Using Deep Neural Networks*, the authors propose Deep Momentum Networks (DMNs) as an approach to momentum strategy that improves on the shortcomings of prior supervised learning models. DMNs are intended to generate positions directly in a way that simultaneously learns both trend estimation and position sizing in the same function.

The authors investigated four common deep neural network architectures - linear regression, multilayer perceptron (MLP), WaveNet, and Long Short-Term Memory (LSTM) - and compared the generated positions to those calculated from existing momentum strategies. In our experimentation, we implemented three of the four network types, omitting WaveNet due to time constraints, and compared our predicted returns with those of the DMNs implemented by the authors.

### CLC Dataset
The data used in the original paper and our experimentation can be accessed through the [Pinnacle Data Corp. CLC Database](https://pinnacledata2.com/clc.html). The database contains ratio-adjusted continuous futures contracts that span a variety of asset classes - including commodities, fixed income, and currency futures. The authors extracted 88 contracts that contained prices from 1990 to 2015. The data is given in CSV format, one for each asset, and formatted as follows:

| Date | Open | High | Low | Close | Open Interest | Volume |
| --- | --- | --- | --- | --- | --- | --- |
| 01/19/2010 | 83.07 | 83.07 | 83.07 | 83.07 | 14 | 145215 |

For our experimentation, we read the contents of the CSVs for a given asset class into numpy arrays in the function `data.GetDataset.get_dataset_by_category()`. For simplicity, we truncated the data, so the features given to the networks only include the data for **Open, High, Low,** and **Close.**

Not all neural networks are capable of inferring information from sequence data. To accommodate for systems with no short term memory, such as a linear model, 5 data-points were aggregated together. The LSTM does need the aggregated data-points since it contains a hidden and cell state which allow it to infer information from a sequence.

### Code
*Enhancing Time Series Momentum Strategies Using Deep Neural Networks*  provides four loss functions: Mean Squared Error (MSE), Binary Cross Entropy (BCE), Loss of the returns, and Sharpe loss ratio. We have provided simple implementations of MSE, Loss of the returns, and Sharpe loss ratio with our code (`src.CustomLoss.py). Since we focused on utilizing our models direct outputs when simulating trading, we have not included an implementation for BCE. Unlike the academic paper our loss functions do not incorporate the volatility of the market.

This code contains three models that have been implemented: a simple linear classifier, a multi-layer perceptron, and a Long Short Term Memory (LSTM) neural networks. Additionally, there is a template for another model as well as some architecture in place to make adding new models as easy as possible. Since all of the models utilize the same hyper-parameters, we can utilize one unified handler for the instantiation and training of the models. 

Let's begin by defining some hyperparameters for our code as well as call the functions provided in `data.GetDataset.py` to create our training and testing sets. Here we will use target lookahead to specify how far into the future we are trying to forecast the stock price for. The input to assets_to_view is a tuple of indices that we'll use to choose which datapoints we'd like to train the models on. For example if the tuple contains (0,2,5) then we will use the open, low, and volume as feature vectors to input to the model.
````python
from src.hyperparameters import *
train_features, train_targets, test_features, test_targets = gd.get_dataset_by_category("commodities",
                                                                                        0.9, aggregate_days=5,
                                                                                        target_lookahead=target_lookahead,
                                                                                        assets_to_view=features,
                                                                                        normalize_data=normalize_data)
````
We've retrieved the dataset, but there are two issues with it. First, it is separated by stock giving it the shape [num_stocks, entries, features] whereas we would like it to have the shape [entries, features]. The second issue is that some of the stocks may not have had the ability to fulfil the aggregation and lookahead parameters, their entries in the dataset must be trimmed.

````python
train_features = [elem for elem in train_features if elem.shape[0] > 0]
train_targets = [elem for elem in train_targets if elem.shape[0] > 0]
train_features = np.concatenate(train_features).astype(np.float32)
train_targets = np.concatenate(train_targets).astype(np.float32)
test_features = [elem for elem in test_features if elem.shape[0] > 0]
test_targets = [elem for elem in test_targets if elem.shape[0] > 0]
````

Now that we have our dataset, we can start exploring the default constructor for the Handler class in `src.Handler.py`.
````python

def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
    self.epochs = epochs
    self.loss_method = loss_method
    self.regularization_method = regularization_method
    self.learning_rate = learning_rate
    self.model = None
    self.batch_size = batch_size
    self.l1enable = l1enable
    self.alpha = alpha
````
These are the hyper-parameters that will be available to the models during training and instantiation. For ease of use, basic implementations of a training and testing loop have been provided here as well.
````python

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

````

Now that we have a mechanism to manage the hyper-parameters and training of our models, I will demonstrate what it would take to implement a simple Linear model based on the one outlined in *Enhancing Time Series Momentum Strategies Using Deep Neural Networks*.

````python
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

````

To finish off the code that we'll need for this model, all we need to do is create Handler class for it an it'll be ready to go.
````python

class LinearHandler(Handler):
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
        super(LinearHandler, self).__init__(epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable, alpha)

    def create_model(self, input_shape, output_shape, dropout):
        self.model = Linear(input_shape, output_shape, dropout)

````

Finally we can get around to creating and training our Linear Classifier. 
````python
linear = LinearHandler(epochs, loss_function, None, 0.01, batch_size, l1enable=regularization)
linear.create_model(train_features.shape[1], 1, dropout)
linear_losses = linear.train(train_features, train_targets)
````

Using the Handler superclass to impliment, instantiat, and train a MLP looks nearly identical, with the only difference being the structure of the model itself. It's implementation has been provided in the `src.MLP.py` code and the class for the model can be seen below.

````python

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
        self.tanh2 = torch.nn.tanh()

    def forward(self, x):
        hidden = self.fc1(x)
        dropout1 = self.dropout1(hidden)
        tanh1 = self.tanh1(dropout1)
        fc2 = self.fc2(tanh1)
        dropout2 = self.dropout2(fc2)
        tanh2 = self.tanh2(dropout2)
        return tanh2

````

The architecture for an LSTM differs greatly from the linear and MLP models. The primary difference is that it is a form of a recurrent neural network, meaning that it feeds some of its parameters back into itself. In the case of the LSTM these parameters are the cell state and the hidden state. These two values give the model the capability to associate information across multiple data-points within a series and learn contextual information. As before, the default constructor will contain the basic structure of the model.
````python
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
````

However from this point on, the implementation differs greatly from the rest of the models. As mentioned, the cell state and the hidden state need to be utilized when computing the forward pass. Additionally, they must be returned by the LSTM unit of the model so that they can be processed in subsequent training iterations.

````python

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

````

Although we can use the same Handler code to manage the hyper-parameters of the model, we'll need to revisit the training loop. The first difference is that the LSTM model trains on sequence data meaning that we'll need to indicate when we're in between sequences with the cell state and the hidden state. To accomplish this we'll need to adjust the shape of the inputs that we're passing into the model back to [stock, num_entries, features]. Documentation has been provided with the training code to help you follow along.

````python

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

````
We dont only need to adjust the outer shape of the dataset, we also need to get entries that have not been aggregated together as input. For that reason we're going to recall the `data.GetDataset.py` code to create a new dataset for training.

````python
train_features, train_targets, test_features, test_targets = gd.get_dataset_by_category("commodities", 0.9,
                                                                                        aggregate_days=1,
                                                                                        target_lookahead=target_lookahead,
                                                                                        assets_to_view=features,
                                                                                        normalize_data=normalize_data)
````

Next we can create and train the model.
````python
lstm = LSTMHandler(epochs, loss_function, None, 0.01, batch_size, l1enable=regularization)
lstm.create_model(train_features[0].shape[1], hidden_parameters, hidden_layers, 1, dropout)
lstm_losses = lstm.train(train_features, train_targets)
````
 
The above code to create each of the models outlined has been provided in the `src` directory which also contains code to call in order to test the models. We'll use that testing code as we discuss evaluating our models. There are two metrics we've chosen from the academic paper to evaluate our models. The first is an auto-trading system that uses the signals retrieved from our models to purchase and sell stocks. The second is a measurement of profitability. Using the auto-trader profitability is the frequency that the model chooses to make a trade that results in a gain.

A very simple auto-trader can be designed using an 'all eggs in one basket' approach. Such a system would simply look at all of the assets available at the current date, pick the asset with the highest expected return, and compute the actual return based on the chosen asset. For simplicities sake, once a transaction has been made, it cannot be reversed and the system must hold the asset until the forecast date is reached. We can also allow the model to choose not to trade for any given day in the event that all transactions would forecast a loss. Since the assets each have a different number of time-steps they are aligned such that they are all present on the final day of trading using the testing set.
 
````python
def model_trading(actual, preds, lookahead):
    """
    A simple all eggs in one basket approach for simulating trading on the market
    :param actual:
    :param preds:
    :param lookahead:
    :return:
    """
    # not every sample is the same size, pad the front with 0s and we'll assume that the stock hadn't existed yet at
    # the beginning of trading
    trades = 0
    profitability = 0
    max_size = 0
    for elem in preds:
        max_size = max(elem.shape[0], max_size)
    # add one additional row to simulate no action
    true_labels = np.zeros((len(actual)+1, max_size))
    true_predictions = np.zeros((len(actual)+1, max_size))
    for i in range(len(actual)):
        current_size = actual[i].shape[0]
        true_labels[i,max_size-current_size:] = np.squeeze(actual[i])
        true_predictions[i, max_size-current_size:] = np.squeeze(preds[i])
    trading_quantity = 1
    trading_route = [1]
    # choose the stocks with the best predictions for each day by getting the index of the highest value of each row
    best_choices = np.argmax(true_predictions, axis=0)
    # for every day in trading
    day = 0
    # Note that this loop will cause variable length outputs, meaning that if a model chooses to do
    # never trade then it will have a number of entries equal to the length of the predictions and if it
    # chooses to make a trade every day then it will be equal to the length of the predictions / lookahead
    while day < best_choices.shape[0] and trading_quantity > 0:
        if true_predictions[best_choices[day], day] == 0:
            day += 1
            trades += 1
        else:
            # get the true return from the targets, not the predictions
            returns = true_labels[best_choices[day], day] if true_labels[best_choices[day], day] > -1 else -1
            trading_quantity += trading_quantity*returns
            # increment the days lookahead, a lookahead of 0 is a special case
            day += lookahead if lookahead > 0 else 1 # special case
            trades += 1
            if returns > 0:
                profitability += 1
        trading_route.append(trading_quantity)
    return np.array(trading_route), profitability/trades
````
 
Using this trading system we can get the predictions by testing each of the models and tracking their performance. Below is some example code of how to test the LSTM, note that the other dataset should be used when testing the Linear and MLP models:
````python
# for each of the assets get the predictions and add them to a list
for i in range(len(test_features)):
    _, pred = lstm.test(test_features[i].astype(np.float32), test_targets[i].astype(np.float32))
    _predictions.append(pred)
lstm_performance, lstm_profitability = model_trading(test_targets, _predictions, lookahead=target_lookahead)

````
Note that if no transaction is made by the model at a time-step, then only one time-step will pass, whereas if a transaction is made by the model then however far the lookahead variable is will be added to the time-steps. What's more, the Linear and MLP models use an aggregated dataset meaning that they'll simply have less data available to them when testing. To plot all of the performances on the same graph, one should linearly interpolate missing values to the shorter models so that they can all be viewed at the same scale.

Once again all of the above code has been provided within the `src/*`, `data/*`, and `main.py` files.

### Replicating our experiments
The hyper-parameters used by our models can be found and changed within the `src.hyperparameters.py` file. Simply running `main.py` will replicate our experiments using the hyper-parameter file. Although we have frozen the random seed, Pytorch does not guarantee complete reproducibility. Below you'll find the hyper-parameters we used to achieve our best results:
````python
batch_size = 512
blocks = ['LSTM', 'linear', 'MLP']
loss_function = 'MSE'
regularization = False
target_lookahead = 2
epochs = 100
features = (0, 1, 2, 3)
normalize_data = True
hidden_parameters = 15
hidden_layers = 1
dropout = 0.5
````


 
### Further Reading
This section includes the original paper and additional resources related to the experiment.
#### Papers
[Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/pdf/1904.04912.pdf), 2019.
#### Posts
[Chapter 8. Recurrent Neural Networks, Dive Into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/index.html)

[Time Series Momentum (aka Trend-Following): A Good Time for a Refresh](https://alphaarchitect.com/2018/02/08/time-series-momentum-aka-trend-following-the-historical-evidence/#:~:text=Time%2Dseries%20momentum%2C%20also%20called,have%20had%20recent%20negative%20returns.)
