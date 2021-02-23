import numpy as np
import json
import os
import pandas as pd


def get_dataset(name, n, t, ratio):
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
        # shuffle by key so that features and targets can be split in unison. Shuffle them so that they're
        # randomly split into training and testing groups independent of time location
        #shuffle_key = np.arange(features.shape[0])
        #np.random.shuffle(shuffle_key)
        if train_x is None:
            train_x = features[:int(ratio * features.shape[0])]
            train_y = targets[:int(ratio * features.shape[0])]
            test_x = features[int(ratio * features.shape[0]):]
            test_y = targets[int(ratio * features.shape[0]):]
        train_x = np.concatenate((train_x, features[:int(ratio*features.shape[0])]))
        train_y = np.concatenate((train_y, targets[:int(ratio*features.shape[0])]))
        test_x = np.concatenate((test_x, features[int(ratio * features.shape[0]):]))
        test_y = np.concatenate((test_y, targets[int(ratio*features.shape[0]):]))
    # shuffle one more time to mix the training set so that we dont train one file at a time.
    # shuffle_key = np.arange(train_x.shape[0])
    # np.random.shuffle(shuffle_key)
    # train_x = train_x[shuffle_key]
    # reshape the labels into column vectors
    # train_y = train_y[shuffle_key].reshape(-1, 1)
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
    # lose one datapoint because I am lazy and to make sure that we have room to look ahead for all points as well
    # as fully a fully populated dataset
    end_itr = data.shape[0]-(n+t)
    for i in range(0, end_itr, n):
        features.append(data[i:i+n].flatten())
        targets.append(normalized_returns(data, i, i+n+t))
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
