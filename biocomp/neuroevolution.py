import math
import random


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


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


class Layer:
    def __init__(self, input_size, target_size, weights=None):
        if not weights:
            weights = [random.uniform(-1, 1) for _ in range(input_size * target_size)]

        self.input_size = input_size
        self.output_size = target_size
        self.weights = weights

    def forward(self, activations):
        if len(activations) != self.input_size:
            raise ValueError(f'Layer expected {self.input_size} input size, '
                             f'found {len(activations)}')
        outputs = [0] * self.output_size
        for i, weight in enumerate(self.weights):
            outputs[i % self.output_size] += weight * activations[i % self.input_size]
        return [sigmoid(value) for value in outputs]


class NeuralNetwork:
    def __init__(self, input_size=4, layers=None):
        self.previous_layer_size = layers[-1].output_size if layers else input_size
        self.layers = layers or []

    def add_layer(self, neuron_count):
        layer = Layer(self.previous_layer_size, neuron_count)
        self.layers.append(layer)
        self.previous_layer_size = neuron_count

    def forward(self, activations):
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations[0]


class EvolutionaryTrainer:

    def __init__(self):
        self.population_size = 100
        self.generation_count = 200
        self.crossover_chance = 0.1
        self.mutation_chance_per_chromosome = 0.5
        self.mutation_chance_per_weight = 0.1
        self.population = []

    def mutate(self):
        for chromosome in self.population:
            if random.uniform(0, 1) > self.mutation_chance_per_chromosome:
                continue

            for layer in chromosome.layers:
                weights = []

                for weight in layer.weights:
                    if random.uniform(0, 1) > self.mutation_chance_per_weight:
                        weights.append(weight)
                        continue

                    weights.append(random.uniform(-1, 1))

                layer.weights = weights

    def train(self, features, labels):
        def create_chromosome():
            feature_count = len(features[0])
            network = NeuralNetwork(feature_count)  # input layer
            network.add_layer(neuron_count=8)  # hidden layer
            network.add_layer(neuron_count=1)  # output layer
            return network

        # initialize population
        self.population = [create_chromosome() for _ in range(self.population_size)]

        for generation in range(self.generation_count):
            # fitness evaluation
            def calculate_error(chromosome):
                error = 0
                for feature, label in zip(features, labels):
                    prediction = chromosome.forward(feature)
                    error += max(0, 1 - prediction if label else prediction)
                return error

            errors = [calculate_error(chromosome) for chromosome in self.population]
            max_error = max(errors)
            print('Min error', min(errors), '\tAverage error', sum(errors) / len(errors))
            population_fitness = [max_error - error for error in errors]
            best_chromosome_index = population_fitness.index(max(population_fitness))
            best_chromosome = self.population[best_chromosome_index]
            total_fitness = sum(population_fitness)

            # selection
            def select_parent():
                target = random.uniform(0, total_fitness)
                partial = 0
                for chromosome, fitness in zip(self.population, population_fitness):
                    partial += fitness
                    if partial > target:
                        return chromosome
                return random.choice(self.population)

            # crossover
            def crossover():
                if self.crossover_chance < random.uniform(0, 1):
                    parent = select_parent()
                    return NeuralNetwork(layers=[
                        Layer(l.input_size, l.output_size, l.weights.copy())
                        for l in parent.layers
                    ])

                mother = select_parent()
                father = select_parent()
                child_layers = []

                for l1, l2 in zip(mother.layers, father.layers):
                    input_size = l1.input_size
                    output_size = l1.output_size
                    split = int(random.uniform(0, len(l1.weights)))
                    weights = l1.weights[:split] + l2.weights[split:]
                    child_layers.append(Layer(input_size, output_size, weights))

                return NeuralNetwork(layers=child_layers)

            self.population = [crossover() for _ in range(self.population_size - 1)]

            # mutation
            self.mutate()

            # retention
            self.population.append(best_chromosome)


def main():
    # dataset_files = {
    #     'data1.txt': parse_binary_string_features,
    #     'data2.txt': parse_binary_string_features,
    #     'data3.txt': parse_floating_point_features,
    # }

    # Set the seed for reproducibility.
    random.seed(1)

    # Grab dataset.
    dataset = load_dataset('data1.txt', parse_binary_string_features)

    # Shuffle and split the dataset into train and test sets.
    random.shuffle(dataset)
    split = int(len(dataset) * 0.75)
    train_x, train_y = zip(*dataset[:split])
    test_x, test_y = zip(*dataset[split:])

    # Train model based on dataset.
    trainer = EvolutionaryTrainer()
    trainer.train(train_x, train_y)


if __name__ == '__main__':
    main()
