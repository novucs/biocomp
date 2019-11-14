from collections import defaultdict

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
    attribute_count = len(features[0])
    is_digital = True
    missing = 0
    occurrences = defaultdict(int)

    for i in range(2 ** attribute_count):
        b = str(bin(i)).split('b')[1]
        b = ("0" * (attribute_count - len(b))) + b
        attributes = [int(e) for e in b]
        attributes_repr = ''.join(map(str, attributes))
        classes = []

        for f, l in zip(features, labels):
            if f == attributes:
                classes.append(l)
                occurrences[attributes_repr] += 1

        if not classes:
            # print('\tMissing:', attributes_repr)
            missing += 1
        elif len(set(classes)) != 1:
            print('\tAttributes:', attributes_repr, 'can be true and false')
            is_digital = False

    if is_digital:
        print('\tThis dataset is a digital multiplexer')
        print(f'\tThere are {missing}/{2 ** attribute_count} missing inputs')
    else:
        print('\tThis dataset is not a digital multiplexer')

    repr_total = sum(occurrences.values())
    repr_max = max(occurrences.values())
    repr_min = min(occurrences.values())
    repr_range = repr_max - repr_min
    max_attrs = next(k for k, v in occurrences.items() if v == repr_max)
    min_attrs = next(k for k, v in occurrences.items() if v == repr_min)
    print('\tRepresentations:')
    print('\t\tTotal:', repr_total, '\tMax:', repr_max, '\tMin:', repr_min, '\tRange:', repr_range)
    print('\t\tMost Represented Attributes:', max_attrs)
    print('\t\tLeast Represented Attributes:', min_attrs)
    print('\t\t0s:', labels.count(0), '\t1s:', labels.count(1))


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
    features, labels, *_ = datasets.split(datasets.load_dataset_4())
    prove_digital_multiplexer(features, labels)
    print()


if __name__ == '__main__':
    main()
