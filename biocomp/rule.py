import itertools
import random

from biocomp import datasets


class GA:
    def __init__(self):
        self.dataset = 'datasets/2019/data1.txt'
        train_x, train_y, *_ = datasets.split(
            datasets.load_dataset(self.dataset, datasets.parse_binary_string_features))
        self.rule_count = len(train_x)
        self.rule_size = len(train_x[0]) + 1
        self.population_size = 50
        self.generation_count = 10000
        self.crossover_chance = 0.5
        # self.mutation_chance = 0.0125
        self.tournament_size = 5
        self.train_x = train_x
        self.train_y = train_y
        self.population = []
        self.overall_best = None
        self.overall_best_fitness = -float('inf')

    @property
    def chromosome_size(self):
        return self.rule_size * self.rule_count

    @property
    def mutation_chance(self):
        return 1 / self.chromosome_size

    def random_chromosome(self, index):
        return random.choice(
            [0, 1, '#'] if index % self.rule_size != (self.rule_size - 1) else [0, 1])

    def load_population(self):
        default_best, default_best_rule_count = None, self.rule_count
        with open('solutions.txt', 'r') as f:
            for line in f:
                tags = {
                    tag.split(':')[0]: ''.join(tag.split(':')[1:])
                    for tag in line.strip().split(' ')
                }

                if tags['dataset'] != self.dataset or default_best_rule_count < int(
                        tags['rule_count']):
                    continue

                default_best = list(
                    map(lambda k: '#' if k == '#' else int(float(k)),
                        tags['rules'].split(',')))
                default_best_rule_count = len(default_best) // self.rule_size

        if not default_best:
            default_best = list(itertools.chain(*[
                list(map(int, tx)) + [int(ty)]
                for tx, ty in zip(self.train_x, self.train_y)
            ]))

        self.rule_count = default_best_rule_count - 1
        self.population = [
            [self.random_chromosome(i) for i in range(self.chromosome_size)]
            for _ in range(self.population_size - 1)
        ]
        self.population.append(default_best[:self.rule_size * self.rule_count])

    def evaluate(self, chromosome, features):
        final_prediction = [0, 0]
        for index in range(0, self.chromosome_size, self.rule_size):
            *rule, prediction = chromosome[index: index + self.rule_size]
            if all(p == f or p == "#" for p, f in zip(rule, features)):
                final_prediction[prediction] += 1
                # return prediction
        return final_prediction.index(max(final_prediction))
        # return 0

    def fitness(self, chromosome, features, labels):
        return sum(1 if self.evaluate(chromosome, f) == l else 0
                   for f, l in zip(features, labels)) / len(labels)

    def train(self):
        self.load_population()

        for generation in range(self.generation_count):
            self.train_step(generation)

    def train_step(self, generation):
        # self.population.append([
        #     0, 0, '#', '#', '#', 1, 1,
        #     1, 1, 1, '#', '#', '#', 1,
        #     0, 1, '#', '#', 1, '#', 1,
        #     1, 0, '#', 1, '#', '#', 1,
        #     '#', '#', '#', '#', '#', '#', 0,
        # ])
        population_fitness = [self.fitness(chromosome, self.train_x, self.train_y)
                              for chromosome in self.population]
        best_fitness = max(population_fitness)
        best_index = population_fitness.index(best_fitness)
        best = self.population[best_index]
        total_fitness = sum(population_fitness)
        mean_fitness = total_fitness / self.population_size

        if best_fitness > self.overall_best_fitness:
            regenerated_population = self.found_new_best(best, best_fitness, generation)
            if regenerated_population:
                return

        print(
            f'Generation: {generation:4} \t'
            f'Best Fitness: {best_fitness:.3f} \t'
            f'Mean Fitness: {mean_fitness:.3f} \t'
            f'Rule Count: {self.rule_count:3} \t'
        )

        def select_parent():
            fittest, fitness = None, -1
            for i in range(self.tournament_size):
                index = int(random.random() * self.population_size)
                if population_fitness[index] > fitness:
                    fitness = population_fitness[index]
                    fittest = self.population[index]
            return fittest

        def crossover():
            mother = select_parent()
            father = select_parent()
            return [random.choice((m, f)) for m, f in zip(mother, father)]

        self.population = [crossover() for _ in range(self.population_size - 1)]

        # mutate
        for chromosome in self.population:
            for index in range(self.chromosome_size):
                if self.mutation_chance > random.random() and \
                        index % self.rule_size != (self.rule_size - 1):
                    chromosome[index] = self.random_chromosome(index)

        self.population.append(self.overall_best)

    def found_new_best(self, best, best_fitness, generation):
        self.overall_best = best
        self.overall_best_fitness = best_fitness

        if best_fitness != 1.0:
            return False

        lines = [f'Found rule in {generation} generations '
                 f'with rule count of {self.rule_count}:']

        with open('solutions.txt', 'a') as f:
            f.write(
                f'dataset:{self.dataset} '
                f'rule_count:{self.rule_count} '
                f'generation:{generation} '
                f'rules:{",".join(map(str, best))}'
                f'\n'
            )

        print('\n'.join(lines))
        self.overall_best = None
        self.overall_best_fitness = -float('inf')
        self.rule_count -= 1
        self.population = [
            [self.random_chromosome(i) for i in range(self.chromosome_size)]
            for _ in range(self.population_size - 1)
        ]
        self.population.append(best[:self.chromosome_size])
        return True


def main():
    ga = GA()
    ga.train()


if __name__ == '__main__':
    main()
