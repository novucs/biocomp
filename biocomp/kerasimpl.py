import random

import numpy
from keras import Sequential
from keras.layers import Dense, SimpleRNN


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


def main():
    # Set the seed for reproducibility.
    random.seed(1)

    # Grab dataset.
    # dataset = load_dataset('data1.txt', parse_binary_string_features)
    # dataset = load_dataset('data2.txt', parse_binary_string_features)
    dataset = load_dataset('data3.txt', parse_floating_point_features)

    # Shuffle and split the dataset into train and test sets.
    random.shuffle(dataset)
    split = int(len(dataset) * 0.75)
    train_x, train_y = zip(*dataset)

    # train_x, train_y = zip(*dataset[:split])
    # test_x, test_y = zip(*dataset[split:])

    def create_model():
        target = Sequential()
        target.add(Dense(8))
        target.add(SimpleRNN(8, input_shape=(len(train_x[0]), 1)))
        target.add(Dense(1))
        target.compile(optimizer='sgd', loss='binary_crossentropy')
        return target

    model = create_model()
    last_loss = None

    for step in range(10000):
        xs = numpy.expand_dims(numpy.array(train_x), axis=2)
        loss = model.train_on_batch(xs, numpy.array(train_y))
        # if loss == last_loss:
        #     # prevents network sometimes getting stuck at 7.712474,
        #     # never learning a thing...
        #     model = create_model()
        last_loss = loss

        print(loss)
        if loss < 0.005:
            break


if __name__ == '__main__':
    main()
