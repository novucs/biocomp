import random

from biocomp import datasets


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
                    pass

            self.population = [crossover() for _ in range(self.population_size - 1)]

            # mutation
            self.mutate()

            # retention
            self.population.append(best_chromosome)


def main():
    # Set the seed for reproducibility.
    random.seed(1)

    # Grab dataset.
    dataset = datasets.load_dataset_1()
    train_x, train_y, test_x, test_y = datasets.split(dataset, 0.9)

    # Train model based on dataset.
    trainer = EvolutionaryTrainer()
    trainer.train(train_x, train_y)


if __name__ == '__main__':
    main()
