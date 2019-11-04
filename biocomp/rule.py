import itertools
import random
from datetime import datetime

from biocomp import datasets


# class Rule:
#     def __init__(self, condition, action):
#         self.condition = condition
#         self.action = action
#
#
# class Individual:
#     def __init__(self):
#         self.rulebase = []
#
#     def crossover(self, other):
#         pass
#
#     def mutate(self):
#         pass


class GA:
    def __init__(self):
        self.dataset = 'datasets/2019/data2.txt'
        train_x, train_y, *_ = datasets.split(
            datasets.load_dataset(self.dataset, datasets.parse_binary_string_features))
        self.rule_count = len(train_x)
        self.rule_size = len(train_x[0]) + 1
        self.population_size = 100
        self.generation_count = 10000
        self.crossover_chance = 0.5
        self.tournament_size = 5
        self.distill_inheritance_chance = 0.33
        self.train_x = train_x
        self.train_y = train_y
        self.population = []
        self.overall_best = None
        self.overall_best_fitness = -float('inf')
        self.generation = 0
        self.checkpoint_fitness = False

    @property
    def chromosome_size(self):
        return self.rule_size * self.rule_count

    @property
    def mutation_chance(self):
        return 1 / self.chromosome_size
        # return 0.01

    def random_chromosome(self, index):
        return random.choice(
            [0, 1, '#'] if index % self.rule_size != (self.rule_size - 1) else [0, 1])

    def load_population(self):
        best, best_rule_count, best_fitness = None, self.rule_count, -float('inf')
        with open('solutions.txt', 'r') as f:
            for line in f:
                tags = {
                    tag.split(':')[0]: ''.join(tag.split(':')[1:])
                    for tag in line.strip().split(' ')
                }

                same_dataset = tags['dataset'] == self.dataset
                better_rule_count = int(tags['rule_count']) < best_rule_count
                equal_rule_count = int(tags['rule_count']) == best_rule_count

                if self.checkpoint_fitness:
                    better_fitness = equal_rule_count and float(
                        tags['fitness']) < best_fitness
                else:
                    better_fitness = better_rule_count and float(tags['fitness']) == 1.0

                if same_dataset and (better_rule_count or better_fitness):
                    best = list(map(lambda k: '#' if k == '#' else int(float(k)),
                                    tags['rules'].split(',')))
                    best_rule_count = len(best) // self.rule_size
                    best_fitness = float(tags['fitness'])

        self.rule_count = best_rule_count - 1 if best_fitness == 1.0 else best_rule_count
        self.generate_population(best, smaller=best is not None and best_fitness == 1.0)

    def generate_population(self, original_best=None, smaller=True):
        if not original_best:
            original_best = list(itertools.chain(*[
                list(map(int, tx)) + [int(ty)]
                for tx, ty in zip(self.train_x, self.train_y)
            ]))

        self.population = [original_best[:self.chromosome_size]]

        if smaller:
            self.population.append(original_best[self.rule_size:])

        for _ in range(self.population_size - (2 if smaller else 1)):
            rules = []

            for index in range(0, self.chromosome_size, self.rule_size):
                if random.random() < self.distill_inheritance_chance:
                    rules.append(original_best[index: index + self.rule_size])
                else:
                    rules.append([self.random_chromosome(i) for i in range(self.rule_size)])

            self.population.append(list(itertools.chain(*rules)))

    def evaluate(self, chromosome, attributes):
        # votes = [0, 0]
        for index in range(0, self.chromosome_size, self.rule_size):
            # print(chromosome)
            *condition, action = chromosome[index: index + self.rule_size]
            if all(p == f or p == "#" for p, f in zip(condition, attributes)):
                # votes[action] += 1
                return action
        # return votes.index(max(votes))
        return None

    def fitness(self, chromosome, features, labels):
        return sum(1 if self.evaluate(chromosome, f) == l else 0
                   for f, l in zip(features, labels)) / len(labels)

    def train(self):
        self.load_population()

        # for generation in range(self.generation_count):
        #     self.generation = generation
        #     self.train_step()

        self.generation = 0
        while True:
            self.generation += 1
            self.train_step()

    def train_step(self):
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
            regenerated_population = self.found_new_best(best, best_fitness)
            if regenerated_population:
                return
            self.save_solution(best, best_fitness)

        if self.generation % 5 == 0:
            print(
                f'Generation: {self.generation:4} \t'
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

            # todo: determine which crossover type is best
            if random.choice((True, False)):
                # Crossover by index
                return [random.choice((m, f)) for m, f in zip(mother, father)]
            else:
                # Crossover by rule
                return list(itertools.chain(*(
                    random.choice((mother[i:i + self.rule_size], father[i:i + self.rule_size]))
                    for i in range(0, self.chromosome_size, self.rule_size)
                )))

        self.population = [crossover() for _ in range(self.population_size - 1)]

        # mutate
        for chromosome in self.population:
            for index in range(self.chromosome_size):
                if self.mutation_chance > random.random() and \
                        index % self.rule_size != (self.rule_size - 1):
                    chromosome[index] = self.random_chromosome(index)

        self.population.append(self.overall_best)

    def found_new_best(self, best, best_fitness):
        self.overall_best = best
        self.overall_best_fitness = best_fitness

        if best_fitness != 1.0:
            return False

        lines = [f'Found rule in {self.generation} generations '
                 f'with rule count of {self.rule_count}:']

        self.save_solution(best, best_fitness)

        print('\n'.join(lines))

        self.overall_best = None
        self.overall_best_fitness = -float('inf')
        self.rule_count -= 1
        self.generate_population(best)
        return True

    def save_solution(self, best, fitness):
        if fitness < 1.0 and not self.checkpoint_fitness:
            return

        with open('solutions.txt', 'a') as f:
            f.write(
                f'dataset:{self.dataset} '
                f'rule_count:{self.rule_count} '
                f'generation:{self.generation} '
                f'fitness:{fitness} '
                f'time:{str(datetime.now()).replace(" ", "_")} '
                f'rules:{",".join(map(str, best))} '
                f'\n'
            )


def main():
    ga = GA()
    ga.train()


if __name__ == '__main__':
    main()
