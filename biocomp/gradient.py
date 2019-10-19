import random

import numpy
from keras import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam

from biocomp import datasets


def main():
    # Set the seed for reproducibility.
    random.seed(1)

    # Grab dataset.
    dataset = datasets.load_dataset_1()
    train_x, train_y, test_x, test_y = datasets.split(dataset, 0.9)

    def create_model():
        target = Sequential()
        # target.add(Dense(4))
        target.add(SimpleRNN(4, input_shape=(len(train_x[0]), 1), use_bias=False, activation='relu'))
        target.add(Dense(1, use_bias=False, activation='relu'))
        target.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy')
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
            print(model.weights)
            break

    # xs = numpy.expand_dims(numpy.array(test_x), axis=2)
    # yhats = model.predict(xs)
    #
    # for features, yhat, yreal in zip(xs, yhats, test_y):
    #     yhat = min(1, max(0, round(yhat[0])))
    #     print(' '.join(str(x[0]) for x in features), ':', yhat, int(yreal), yreal == yhat)


if __name__ == '__main__':
    main()
