import math
import random
import keras
import numpy
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam


def parse_binary_string_features(features):
    return [float(i) for i in features[0]]


def parse_floating_point_features(features):
    return [float(i) for i in features]


def load_dataset(filename, parse_features):
    with open(filename, 'r') as file:
        contents = [line.split() for line in file.readlines()[1:] if line]

    if len(contents) == 0:
        raise ValueError('Invalid dataset, no data points file contents')

    features = [parse_features(values[:-1]) for values in contents]
    labels = [bool(int(values[-1])) for values in contents]
    return list(zip(features, labels))


def main():
    # Set the seed for reproducibility.
    random.seed(1)

    # Grab dataset.
    dataset = load_dataset('data1.txt', parse_binary_string_features)
    # dataset = load_dataset('data2.txt', parse_binary_string_features)
    # dataset = load_dataset('data3.txt', parse_floating_point_features)

    # Shuffle and split the dataset into train and test sets.
    random.shuffle(dataset)
    split = int(len(dataset) * 0.75)
    train_x, train_y = zip(*dataset)
    # train_x, train_y = zip(*dataset[:split])
    # test_x, test_y = zip(*dataset[split:])

    model = Sequential()
    model.add(Dense(32))
    model.add(LSTM(32, input_shape=(len(train_x[0]), 1)))
    model.add(Dense(32))
    # model.add(Dense(32))
    # model.add(Dense(32))
    # model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # for x, y in zip(train_x, train_y):
    #     print(''.join([str(int(v)) for v in x]), int(y))

    for step in range(1000):
        xs = numpy.expand_dims(numpy.array(train_x), axis=2)
        # xs = numpy.array(train_x)
        loss = model.train_on_batch(xs, numpy.array(train_y))
        print(loss)
    # for xc, x, y in zip(train_x, model.predict(numpy.expand_dims(numpy.array(train_x), axis=2)), train_y):
    #     print(''.join([str(int(v)) for v in xc]), int(min(1, max(0, round(x[0])))), int(y))


if __name__ == '__main__':
    main()
