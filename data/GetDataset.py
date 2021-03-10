import numpy as np
import json
import os
import pandas as pd


def get_aggregated_dataset(name, n, t, ratio):
    """
    Retrieve the dataset with name "name". Concatenate information over N trading days and associate targets over
    N+T days where T is how far in the future we are looking to predict returns. The dataset ratio is determined
    by the input param ratio, where ratio represents the percent of training samples and 1-ratio is the number of test
    samples.
    :param string name: the name of the dataset: [commodities, equities, fixed income, fx]
    :param int n: the number of days to consider for each training sample
    :param int t: the number of days in the future to consider for the label
    :param float ratio: the ratio to use when splitting the data into train and test
    :return: trainX, testX, trainY,  testY
    """
    np.random.seed(0)
    assert name in ["commodities", "equities", "fixed income", "fx"], \
        f'name must be a valid entry within the dataset: {["commodities", "equities", "fixed income", "fx"]}'
    # get files in dataset
    with open("data/datasets.json", "r") as f:
        dataset = [elem+"_" for elem in json.loads(f.read())[name]]
    dataset = get_matching_files(dataset)
    # instantiate array values
    train_x, train_y, test_x, test_y = None, None, None, None
    # for each file
    for f in dataset:
        # get the features and targets from the file
        features, targets = parse_file(f, n, t)
        if features.shape[0]:
            if train_x is None:
                train_x = features[:int(ratio * features.shape[0])]
                train_y = targets[:int(ratio * features.shape[0])]
                test_x = features[int(ratio * features.shape[0]):]
                test_y = targets[int(ratio * features.shape[0]):]
            train_x = np.concatenate((train_x, features[:int(ratio*features.shape[0])]))
            train_y = np.concatenate((train_y, targets[:int(ratio*features.shape[0])]))
            test_x = np.concatenate((test_x, features[int(ratio * features.shape[0]):]))
            test_y = np.concatenate((test_y, targets[int(ratio*features.shape[0]):]))
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    return train_x.astype(np.float32), train_y.astype(np.float32), test_x.astype(np.float32), test_y.astype(np.float32)


def get_matching_files(dataset):
    """
    get all filenames that match the target dataset
    :param dataset: target to match
    :return: list of filenames
    """
    all_data = os.listdir("data/CLCDATA")
    target_dataset = []
    for files in all_data:
        for f in dataset:
            if files.startswith(f):
                target_dataset.append(files)
    return target_dataset


def parse_file(filename, n, t):
    """
    create datapoints from the file. If there is any overlap in the data, then one risks mixing the training
    and testing data.
    :param string filename: the name of the file containing the data
    :param n: the n datapoints to use when creating a datapoint
    :param t: the number of points to look ahead for the label
    :return: features, targets
    """
    data = pd.read_csv(f"data/CLCDATA/{filename}", header=None).to_numpy()
    # dont need dates, they can be trimmed
    data = data[:, 1:]
    features = []
    targets = []
    # lose one datapoint to make sure that we have room to look ahead for all points as well
    # as fully a fully populated dataset
    end_itr = data.shape[0]-(n+t)
    for i in range(0, end_itr, n):
        features.append(data[i:i+n].flatten())
        targets.append(normalized_returns(data, i, i+n+t))

    mask = np.array([1, 1, 1, 1, 0, 0]*n)
    start_index = 0
    while start_index < len(features) and np.all(features[start_index][mask] == 0.0):
        start_index += 1
    features = features[start_index:]
    features = np.array(features)
    targets = targets[start_index:]
    targets = np.array(targets)

    if features.shape[0] > 0:
        features = features[:, mask]/np.max(features[:, mask])
        targets = targets/np.abs(targets).max()

    return np.array(features), np.array(targets)


def normalized_returns(datapoints, s, e):
    """
    compute the normalized returns for the asset: formula given as r_start_to_end/(sigma_t * sqrt(s+t))
    :param ndarray datapoints: the datapoints in the current dataset
    datapoints are formatted as open, high, low, close, open interest, volume
    r_n,n+1 = end_open - start_open
    :param int s: the start index
    :param int e: the end index
    :return: number value representing the normalized returns
    """
    start = datapoints[s]
    end = datapoints[e]
    # esimated returns = sell open - buy open
    r_start_to_end = end[0] - start[0]
    # sqrt of time frame
    timeframe = np.sqrt(e-s)
    # std_dev of timeframe
    sigma_t = datapoints[s:e][0, :].std()
    return r_start_to_end/((sigma_t+1e-26)*timeframe)


def get_normal_dataset(name, ratio, target_lookahead, normalize=True):
    """
    Get the dataset without grouping elements togehter.
    :param name: name of the dataset
    :param ratio: split of the data
    :param target_lookahead: how far to lookahead when creating a label
    :return: trainX, testX, trainY, testY
    """
    np.random.seed(0)
    assert name in ["commodities", "equities", "fixed income", "fx"], \
        f'name must be a valid entry within the dataset: {["commodities", "equities", "fixed income", "fx"]}'
    # get files in dataset
    with open("data/datasets.json", "r") as f:
        dataset = [elem + "_" for elem in json.loads(f.read())[name]]
    dataset = get_matching_files(dataset)
    # instantiate array values
    train_x, train_y, test_x, test_y = None, None, None, None
    for f in dataset:
        features, targets = parse_file(f, 1, target_lookahead)
        if features.shape[0]:
            if train_x is None:
                train_x = features[:int(ratio * features.shape[0])]
                train_y = targets[:int(ratio * features.shape[0])]
                test_x = features[int(ratio * features.shape[0]):]
                test_y = targets[int(ratio * features.shape[0]):]
            else:
                train_x = np.concatenate((train_x, features[:int(ratio*features.shape[0])]))
                train_y = np.concatenate((train_y, targets[:int(ratio*features.shape[0])]))
                test_x = np.concatenate((test_x, features[int(ratio * features.shape[0]):]))
                test_y = np.concatenate((test_y, targets[int(ratio*features.shape[0]):]))
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    return train_x.astype(np.float32), train_y.astype(np.float32), test_x.astype(np.float32), test_y.astype(np.float32)


def get_dataset_by_category(name, ratio, method='normalized_returns', target_lookahead=2, aggregate_days=5):
    """
    Split the dataset into categories where each category represents a different company. This split uses the simple
    features and labels
    :param name: name of the dataset
    :param ratio: split of the data
    :return:
    """
    np.random.seed(0)
    assert name in ["commodities", "equities", "fixed income", "fx"], \
        f'name must be a valid entry within the dataset: {["commodities", "equities", "fixed income", "fx"]}'
    # get files in dataset
    with open("data/datasets.json", "r") as f:
        dataset = [elem + "_" for elem in json.loads(f.read())[name]]
    dataset = get_matching_files(dataset)
    # instantiate array values
    train_x, train_y, test_x, test_y = None, None, None, None
    for i in range(len(dataset)):
        if method == "normalized_returns":
            features, targets = parse_file(dataset[i], aggregate_days, target_lookahead)
        elif method == "sanity_check":
            features, targets = parse_raw_features_targets(dataset[i])
        if train_x is None:
            train_x = [features[:int(ratio * features.shape[0])]]
            train_y = [targets[:int(ratio * features.shape[0])]]
            test_x = [features[int(ratio * features.shape[0]):]]
            test_y = [targets[int(ratio * features.shape[0]):]]
        else:
            train_x.append(features[:int(ratio * features.shape[0])])
            train_y.append(targets[:int(ratio * features.shape[0])])
            test_x.append(features[int(ratio * features.shape[0]):])
            test_y.append(targets[int(ratio * features.shape[0]):])
        train_y[i] = train_y[i].reshape(-1, 1)
        test_y[i] = test_y[i].reshape(-1, 1)
    #train_x = np.array(train_x)
    #train_y = np.array(train_y)
    #test_x = np.array(test_x)
    #test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y


def parse_raw_features_targets(f):
    """
    A sanity check for the dataset, take a file and return a more simple regressional dataset split, simply try to
    predict the volume for the day
    :param f: filename
    :return: features, targets
    """
    data = pd.read_csv(f"data/CLCDATA/{f}", header=None).to_numpy()
    # dont need dates, they can be trimmed
    data = data[:, 1:]
    # let the features be the data and the targets be the volume,
    # to keep it from getting too easy lets remove open interest
    # the features are open, high, low, close
    features = data[:, :4]
    # targets are volume
    targets = data[:, -1]
    return np.array(features), np.array(targets)
