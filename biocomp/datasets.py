import random


def parse_binary_string_features(features):
    return [float(i) for i in features[0]]


def parse_floating_point_features(features):
    return [round(float(i)) for i in features]
    # return [float(i) for i in features]


def load_dataset(filename, parse_features):
    with open(filename, 'r') as file:
        contents = [line.split() for line in file.readlines()[1:] if line]

    if len(contents) == 0:
        raise ValueError('Invalid dataset, no data points file contents')

    features = [parse_features(values[:-1]) for values in contents]
    labels = [bool(int(values[-1])) for values in contents]
    return list(zip(features, labels))


def load_dataset_1():
    return load_dataset('data1.txt', parse_binary_string_features)


def load_dataset_2():
    return load_dataset('data2.txt', parse_binary_string_features)


def load_dataset_3():
    return load_dataset('data3.txt', parse_floating_point_features)


def split(dataset, train_percent=1.0):
    if train_percent >= 1.0:
        train_x, train_y = zip(*dataset)
        return train_x, train_y

    if train_percent <= 0.0:
        test_x, test_y = zip(*dataset)
        return test_x, test_y

    random.shuffle(dataset)
    index = int(len(dataset) * train_percent)
    train_x, train_y = zip(*dataset[:index])
    test_x, test_y = zip(*dataset[index:])
    return train_x, train_y, test_x, test_y
