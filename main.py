import data.GetDataset as gd
from src import *
import torch
import numpy as np
import matplotlib.pyplot as plt


def model_trading(actual, preds, lookahead):
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
    while day < best_choices.shape[0] and trading_quantity > 0:
        if true_predictions[best_choices[day], day] == 0:
            day += 1
            trades += 1
        else:
            returns = true_labels[best_choices[day], day] if true_labels[best_choices[day], day] > -1 else -1
            trading_quantity += trading_quantity*returns
            day += lookahead if lookahead > 0 else 1 # special case
            trades += 1
            if returns > 0:
                profitability += 1
        trading_route.append(trading_quantity)
    return np.array(trading_route), profitability/trades


if __name__ == '__main__':
    from src.hyperparameters import *
    plt_shape = []
    prof_values = []
    if 'linear' in blocks or 'MLP' in blocks:
        train_features, train_targets, test_features, test_targets = gd.get_dataset_by_category("commodities",
                                                                                                0.9, aggregate_days=5,
                                                                                                target_lookahead=target_lookahead,
                                                                                                assets_to_view=features,
                                                                                                normalize_data=normalize_data)
        train_features = [elem for elem in train_features if elem.shape[0] > 0]
        train_targets = [elem for elem in train_targets if elem.shape[0] > 0]
        train_features = np.concatenate(train_features).astype(np.float32)
        train_targets = np.concatenate(train_targets).astype(np.float32)
        test_features = [elem for elem in test_features if elem.shape[0] > 0]
        test_targets = [elem for elem in test_targets if elem.shape[0] > 0]
    if 'linear' in blocks:
        print('----- Linear -----')
        linear = LinearHandler(epochs, loss_function, None, 0.01, batch_size, l1enable=regularization)
        linear.create_model(train_features.shape[1], 1)
        linear_losses = linear.train(train_features, train_targets)
        _predictions = []
        for i in range(len(test_features)):
            _, pred = linear.test(test_features[i].astype(np.float32), test_targets[i].astype(np.float32))
            _predictions.append(pred.detach().numpy())
        linear_performance, linear_profitability = model_trading(test_targets, _predictions, lookahead=target_lookahead)
        plt_shape.append(linear_performance.shape[0])
        prof_values.append(linear_profitability)
    if 'MLP' in blocks:
        print('\n\n\n----- MLP -----')
        mlp = MLPHandler(epochs, loss_function, None, 0.01, batch_size, l1enable=regularization)
        # they never specify the number of hidden nodes
        # page 6: "a 2-layer neural network can be used to incorporate non-linear effects."
        # Looks like just 1 hidden + 1 output
        mlp.create_model(train_features.shape[1], 15, 1)
        mlp_losses = mlp.train(train_features, np.squeeze(train_targets))
        _predictions = []
        for i in range(len(test_features)):
            _, pred = mlp.test(test_features[i].astype(np.float32), np.squeeze(test_targets[i].astype(np.float32)))
            _predictions.append(pred.detach().numpy())
        mlp_performance, mlp_profitability = model_trading(test_targets, _predictions, lookahead=target_lookahead)
        plt_shape.append(mlp_performance.shape[0])
        prof_values.append(mlp_profitability)

    if 'LSTM' in blocks:
        _predictions = []
        train_features, train_targets, test_features, test_targets = gd.get_dataset_by_category("commodities", 0.9,
                                                                                                aggregate_days=1,
                                                                                                target_lookahead=target_lookahead,
                                                                                                assets_to_view=features,
                                                                                                normalize_data=normalize_data)
        # aggregate the training set together, no need to differentiate between the different sets during training
        print('\n\n\n----- LSTM -----')
        lstm = LSTMHandler(epochs, loss_function, None, 0.01, batch_size, l1enable=regularization)
        lstm.create_model(train_features[0].shape[1], 15, 1, 1)
        lstm_losses = lstm.train(train_features, train_targets)
        for i in range(len(test_features)):
            _, pred = lstm.test(test_features[i].astype(np.float32), test_targets[i].astype(np.float32))
            _predictions.append(pred)
        lstm_performance, lstm_profitability = model_trading(test_targets, _predictions, lookahead=target_lookahead)
        best_possible_performance = model_trading(test_targets, test_targets, lookahead=2)
        plt_shape.append(lstm_performance.shape[0])
        prof_values.append(lstm_profitability)

    max_size = max(plt_shape)
    if "LSTM" in blocks:
        lstm_key = np.linspace(0, lstm_performance.shape[0], max_size)
        lstm_performance = np.interp(lstm_key, np.arange(lstm_performance.shape[0]), lstm_performance)
        plt.plot(lstm_performance, c='r', label='LSTM')
    if "linear" in blocks:
        linear_key = np.linspace(0, linear_performance.shape[0], max_size)
        linear_performance = np.interp(linear_key, np.arange(linear_performance.shape[0]), linear_performance)
        plt.plot(linear_performance, c='b', label='Linear')
    if "MLP" in blocks:
        mlp_key = np.linspace(0, mlp_performance.shape[0], max_size)
        mlp_performance = np.interp(mlp_key, np.arange(mlp_performance.shape[0]), mlp_performance)
        plt.plot(mlp_performance, c='g', label='MLP')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Growth Ratio')
    plt.title(f'Cumulative Returns - {loss_function} Loss')
    plt.show()

    prof_bars = blocks
    plt.xticks(range(len(blocks)), prof_bars)
    plt.xlabel('Model')
    plt.ylabel("Profitability")
    plt.title(f"Profitability of the Models - {loss_function}")
    plt.bar(range(len(blocks)), prof_values)
    plt.show()