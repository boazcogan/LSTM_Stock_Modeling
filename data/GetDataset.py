import numpy as np
import json
import os
import pandas as pd


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


def parse_file(filename, n, t, data_to_use, normalize_data=True):
    """
    create datapoints from the file. If there is any overlap in the data, then one risks mixing the training
    and testing data.
    :param string filename: the name of the file containing the data
    :param n: the n datapoints to use when creating a datapoint
    :param t: the number of points to look ahead for the label
    :param data_to_use: the cells per item to include in the dataset
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

    # define a mask to extract only the points we would like to use for training
    data_to_use = np.array(data_to_use)
    mask = np.zeros(data.shape[1]).astype(bool)
    mask[data_to_use] = 1
    mask = mask.tolist()*n

    for i in range(0, end_itr, n):
        current_item = data[i:i+n].flatten()
        current_item = current_item[mask]
        features.append(current_item)
        # targets is based on the open for each day, therefore no need to apply the mask
        targets.append(normalized_returns(data, i, i+n+t))

    # trim away all entries before the asset is available for trading
    start_index = 0
    while start_index < len(features) and np.all(features[start_index] == 0.0):
        start_index += 1
    features = features[start_index:]
    features = np.array(features)
    targets = targets[start_index:]
    targets = np.array(targets)

    # if there are any features, then normalize them to be between 0 and 1
    if features.shape[0] > 0 and normalize_data:
        features = features/np.max(features)
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


def get_dataset_by_category(name, ratio, method='normalized_returns', target_lookahead=2, aggregate_days=5,
                            assets_to_view=(0, 1, 2, 3, 4, 5), normalize_data=True):
    """
    Retrieve the dataset with name "name". Concatenate information over N trading days and associate targets over
    N+T days where T is how far in the future we are looking to predict returns. The dataset ratio is determined
    by the input param ratio, where ratio represents the percent of training samples and 1-ratio is the number of test
    samples.
    :param normalize_data: Normalize the feature vectors
    :param aggregate_days: the number of days to consider for each training sample
    :param target_lookahead: the number of days in the future to consider for the label
    :param method: The type of target to be parsed
    :param string name: the name of the dataset: [commodities, equities, fixed income, fx]
    :param float ratio: the ratio to use when splitting the data into train and test
    :param tuple assets_to_view: Select the asset metadata indices to include in each training sample
    :return: trainX, testX, trainY,  testY

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
            features, targets = parse_file(dataset[i], aggregate_days, target_lookahead,
                                           data_to_use=assets_to_view, normalize_data=normalize_data)
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
