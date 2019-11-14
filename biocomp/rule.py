import random
from datetime import datetime
from typing import (
    List,
    Union,
)

from biocomp import datasets


class Rule:
    def __init__(self, ga, condition, action):
        self.ga: GA = ga
        self.condition: List[Union[List[float, float], str]] = condition
        self.action: int = action

    @property
    def generalisation(self):
        return self.condition.count('#') / len(self.condition)

    def uniform_crossover(self, other):
        condition = [random.choice((s, o)) for s, o in zip(self.condition, other.condition)]
        action = random.choice((self.action, other.action))
        return Rule(self.ga, condition, action)

    def mutate_bounds(self, bounds):
        if bounds == '#':
            return self.random_bounds() if random.random() < self.ga.mutation_chance else bounds

        lower, upper = bounds
        lower = Rule.random_bound(upper=upper) if random.random() < self.ga.mutation_chance else lower
        upper = Rule.random_bound(lower=lower) if random.random() < self.ga.mutation_chance else upper
        return lower, upper

    def mutate_action(self, action):
        return action if self.ga.mutation_chance < random.random() else random.choice((0, 1))

    def mutate(self):
        condition = list(map(self.mutate_bounds, self.condition))
        action = self.mutate_action(self.action)
        return Rule(self.ga, condition, action)

    def matches(self, features):
        return all(c == '#' or c[0] < f < c[1] for c, f in zip(self.condition, features))

    def copy(self):
        return Rule(self.ga, self.condition.copy(), self.action)

    @staticmethod
    def random_bound(lower=None, upper=None):
        lower = lower if lower is not None else -0.25
        upper = upper if upper is not None else 1.25
        return random.uniform(lower, upper)

    @staticmethod
    def random_bounds(surrounding=None):
        lower = Rule.random_bound(upper=surrounding)
        upper = Rule.random_bound(lower=max(lower, surrounding or 0))
        return lower, upper

    @staticmethod
    def generate(ga):
        condition = random.choices((Rule.random_bounds(), '#'), k=ga.condition_size)
        action = random.choice((0, 1))
        return Rule(ga, condition, action)

    @staticmethod
    def load(ga, dump):
        *condition, action = dump.split(',')
        condition = [c if c == '#' else list(map(float, c.split('~'))) for c in condition]
        action = int(action)
        return Rule(ga, condition, action)

    def dump(self):
        condition = [c if c == '#' else f'{c[0]}~{c[1]}' for c in self.condition]
        return ','.join(map(str, (condition + [self.action])))

    @staticmethod
    def from_sample(ga, features, label):
        return Rule(ga, list(map(Rule.random_bounds, features)), int(label))

    def subsumes(self, other):
        return all(s == '#' or (o != '#' and s[0] <= o[0] and o[1] <= s[1])
                   for s, o in zip(self.condition, other.condition))


class Individual:
    def __init__(self, ga, rules=None):
        self.ga: GA = ga
        self.rules: List[Rule] = rules or []

    @property
    def generalisation(self):
        return sum(r.generalisation for r in self.rules) / len(self.rules)

    @property
    def rule_count(self):
        return len(self.rules)

    def uniform_crossover(self, other):
        rules = [s.uniform_crossover(o) for s, o in zip(self.rules, other.rules)]
        return Individual(self.ga, rules)

    def crossover_by_rule(self, other):
        rules = [random.choice((s, o)) for s, o in zip(self.rules, other.rules)]
        return Individual(self.ga, rules)

    def crossover(self, other):
        # todo: determine which crossover type is best
        return self.uniform_crossover(other) if random.choice((True, False)) \
            else self.crossover_by_rule(other)

    def mutate(self):
        # mutate each rule individually
        rules = [r.mutate() for r in self.rules]

        # randomly swap order of rules
        for i in range(len(rules) - 1):
            if random.random() < self.ga.mutation_chance:
                rules[i], rules[i + 1] = rules[i + 1], rules[i]

        return Individual(self.ga, rules)

    def evaluate(self, features):
        return next((r.action for r in self.rules if r.matches(features)), None)

    def correct_count(self, features, labels):
        return sum(int(self.evaluate(f) == l) for f, l in zip(features, labels))

    def fitness(self, features, labels):
        correctness = self.correct_count(features, labels) / len(labels)
        generalisation = self.generalisation / len(labels)
        # generalisation = 0
        return correctness + generalisation

    def wrong_classifications(self, features, labels):
        return [(f, l) for f, l in zip(features, labels) if self.evaluate(f) != l]

    def copy(self):
        return Individual(self.ga, self.rules.copy())

    @staticmethod
    def generate(ga):
        rules = [Rule.generate(ga) for _ in range(ga.rule_count)]
        return Individual(ga, rules)

    @staticmethod
    def load(ga, dump, rule_count):
        params = dump.split(',')
        rules = [
            Rule.load(ga, ','.join(params[i:i + ga.rule_size]))
            for i in range(0, ga.rule_size * rule_count, ga.rule_size)
        ]
        return Individual(ga, rules)

    def dump(self):
        return ','.join(r.dump() for r in self.rules)

    @staticmethod
    def from_samples(ga, features, labels):
        if ga.rule_count == len(labels):
            samples = zip(features, labels)
        else:
            samples = random.choices(list(zip(features, labels)), k=ga.rule_count)
        rules = [Rule.from_sample(ga, f, l) for f, l in samples]
        return Individual(ga, rules)

    def remove_rule(self):
        # todo: remove based on niche
        individual = self.copy()
        individual.rules.remove(random.choice(self.rules))
        return individual

    def compress(self):
        rules = [
            rule.copy()
            for i, rule in enumerate(self.rules)
            if not any(o.subsumes(rule) for o in self.rules[:i])
        ]
        return Individual(self.ga, rules)

    def cover(self, features, labels):
        wrong = self.wrong_classifications(features, labels)
        rules = self.compress().rules
        rules = [Rule.from_sample(self.ga, f, l) for f, l in wrong[:self.ga.rule_count - len(rules)]] + rules
        samples = random.choices(list(zip(features, labels)), k=(self.ga.rule_count - len(rules)))
        rules += [Rule.from_sample(self.ga, f, l) for f, l in samples]
        return Individual(self.ga, rules)


class GA:
    def __init__(self):
        self.dataset = 'datasets/2019/data2.txt'
        train_x, train_y, *_ = datasets.split(
            datasets.load_dataset(self.dataset, datasets.parse_binary_string_features))

        self.rule_count = len(train_x)
        self.condition_size = len(train_x[0])
        self.rule_size = len(train_x[0]) + 1
        self.population_size = 100
        self.generation_count = 10000
        self.crossover_chance = 0.5
        self.tournament_size = 5
        self.distill_inheritance_chance = 0.33
        self.train_x = train_x
        self.train_y = train_y
        self.population: List[Individual] = []
        self.overall_best = None
        self.overall_best_fitness = -float('inf')
        self.generation = 0
        self.checkpoint_fitness = False
        self.cover_chance = 0.1

        self.fitness_threshold = (59 / 60)
        # self.fitness_threshold = 1.0
        self.noisy_prints = False

        self.alternatives = set()

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
        best, best_rule_count, best_fitness = None, float('inf'), -float('inf')

        with open('solutions.txt', 'r') as f:
            for line in f:
                tags = {
                    tag.split(':')[0]: ''.join(tag.split(':')[1:])
                    for tag in line.strip().split(' ')
                }

                same_dataset = tags['dataset'] == self.dataset
                rule_count = int(tags['rule_count'])
                better_rule_count = rule_count < best_rule_count
                equal_rule_count = rule_count == best_rule_count

                if self.checkpoint_fitness:
                    better_fitness = equal_rule_count and float(tags['fitness']) < best_fitness
                else:
                    better_fitness = better_rule_count and float(tags['fitness']) >= self.fitness_threshold

                if same_dataset and (better_rule_count or better_fitness):
                    best = Individual.load(self, tags['rules'], rule_count)
                    best_fitness = float(tags['fitness'])
                    self.rule_count = best_rule_count = rule_count

        reduce_rule_count = best_fitness >= self.fitness_threshold
        if reduce_rule_count:
            self.rule_count -= 1

        self.generate_population(best, smaller=reduce_rule_count)

    def sample_from_dataset(self):
        index = random.randint(0, len(self.train_x) - 1)
        return self.train_x[index], self.train_y[index]

    def generate_population(self, best=None, smaller=True):
        self.population = self.covered_population() if not best \
            else self.reduced_population(best) if smaller \
            else self.similar_population(best)

    def similar_population(self, best):
        return [best.copy()] + self.covered_population()[:-1]

    def reduced_population(self, best):
        return [
            best.remove_rule() if random.random() < self.distill_inheritance_chance
            else Individual.from_samples(self, self.train_x, self.train_y)
            for _ in range(self.population_size)
        ]

    def covered_population(self):
        return [
            Individual.from_samples(self, self.train_x, self.train_y)
            for _ in range(self.population_size)
        ]

    def tournament_selection(self, population_fitness):
        fittest, fitness = None, -1
        for i in range(self.tournament_size):
            index = int(random.random() * self.population_size)
            if population_fitness[index] > fitness:
                fitness = population_fitness[index]
                fittest = self.population[index]
        return fittest

    def create_offspring(self, population_fitness):
        mum = self.tournament_selection(population_fitness)
        dad = self.tournament_selection(population_fitness)
        offspring = mum.crossover(dad)
        offspring = offspring.mutate()
        if random.random() < self.cover_chance:
            offspring = offspring.cover(self.train_x, self.train_y)
        return offspring

    def train(self):
        self.load_population()
        self.generation = 0
        while True:
            self.generation += 1
            self.train_step()

    def train_step(self):
        population_fitness = [individual.fitness(self.train_x, self.train_y)
                              for individual in self.population]
        best_fitness = max(population_fitness)
        best_index = population_fitness.index(best_fitness)
        best = self.population[best_index].cover(self.train_x, self.train_y)
        total_fitness = sum(population_fitness)
        mean_fitness = total_fitness / self.population_size

        if self.noisy_prints:
            print(best.wrong_classifications(self.train_x, self.train_y))

        if best_fitness == self.overall_best_fitness:
            self.overall_best = best
            self.overall_best_fitness = best_fitness

            if str(best) not in self.alternatives:
                self.alternatives.add(str(best))
                self.checkpoint_fitness = True
                self.save_solution(best, best_fitness, 'alternatives.txt')
                self.checkpoint_fitness = False
            elif random.random() < self.mutation_chance:
                self.load_population()

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

        self.population = [self.create_offspring(population_fitness) for _ in range(self.population_size - 1)]
        self.population.append(self.overall_best)

    def found_new_best(self, best, best_fitness):
        if best_fitness < self.fitness_threshold:
            self.overall_best = best
            self.overall_best_fitness = best_fitness
            return False

        best = best.compress()
        self.rule_count = best.rule_count

        lines = [f'Found rule in {self.generation} generations '
                 f'with rule count of {self.rule_count}:']

        self.save_solution(best, best_fitness)

        print('\n'.join(lines))

        self.overall_best = None
        self.overall_best_fitness = -float('inf')
        self.rule_count -= 1
        self.generate_population(best)
        return True

    def save_solution(self, best, fitness, file='solutions.txt'):
        if fitness < self.fitness_threshold and not self.checkpoint_fitness:
            return

        with open(file, 'a') as f:
            f.write(
                f'dataset:{self.dataset} '
                f'rule_count:{self.rule_count} '
                f'generation:{self.generation} '
                f'fitness:{fitness} '
                f'time:{str(datetime.now()).replace(" ", "_")} '
                f'rules:{best.dump()}'
                f'\n'
            )


def main():
    ga = GA()
    ga.train()


if __name__ == '__main__':
    main()
