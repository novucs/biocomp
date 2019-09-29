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


class EvolutionaryTrainer:

    def __init__(self):
        self.population_size = 100
        self.generation_count = 200
        self.crossover_chance = 0.1
        self.mutation_chance_per_chromosome = 0.5
        self.mutation_chance_per_weight = 0.1
        self.dimensions = 3
        self.population = []

    def predict(self, chromosome, features):
        prediction = chromosome[-1]
        powers = [i + 1 for i in range(self.dimensions)] * len(features)
        for scale, feature, power in zip(chromosome, features, powers):
            prediction += scale * (feature ** power)
        return prediction

    def create_chromosome(self):
        return [random.uniform(0, 1) for _ in range(self.dimensions + 1)]

    def train(self, features, labels):

        # initialize population
        self.population = [self.create_chromosome() for _ in range(self.population_size)]

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
