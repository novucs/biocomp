from biocomp import datasets


# missing in data1:
# 000011
# 011100
# 100010
# 110100

# missing in data2:
# 000001
# 001110
# 101001
# 111111
# looks like 011000 is always incorrectly predicted,
# must be an anomaly in the dataset

# dataset 3 is a digital multiplexer with noise, no missing values

def what_is_missing(features):
    for i in range(2 ** 6):
        b = str(bin(i)).split('b')[1]
        b = ("0" * (6 - len(b))) + b
        attributes = [float(int(e)) for e in b]
        if attributes not in features:
            print(f'\tMissing: {b}')


def prove_digital_multiplexer(features, labels):
    features = [[round(j) for j in i] for i in features]
    is_digital = True
    missing = False

    for i in range(2 ** 6):
        b = str(bin(i)).split('b')[1]
        b = ("0" * (6 - len(b))) + b
        attributes = [float(int(e)) for e in b]
        classes = []

        for f, l in zip(features, labels):
            if f == attributes:
                classes.append(l)

        if not classes:
            print('missing:', attributes)
            missing = True
        elif len(set(classes)) != 1:
            print('attributes:', attributes, 'can be true and false')
            is_digital = False

    if not missing:
        print('\tAll possible inputs are in this dataset')

    if is_digital:
        print('\tThis dataset is a digital multiplexer')


def main():
    print('Processing dataset 1:')
    features, labels, *_ = datasets.split(datasets.load_dataset_1())
    what_is_missing(features)
    print()

    print('Processing dataset 2:')
    features, labels, *_ = datasets.split(datasets.load_dataset_2())
    what_is_missing(features)
    print()

    print('Processing dataset 3:')
    features, labels, *_ = datasets.split(datasets.load_dataset_3())
    prove_digital_multiplexer(features, labels)
    print()

    print('Processing dataset 4:')
    features, labels, *_ = datasets.split(datasets.load_dataset_3())
    prove_digital_multiplexer(features, labels)
    print()


if __name__ == '__main__':
    main()
