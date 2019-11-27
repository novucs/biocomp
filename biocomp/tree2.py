import itertools
import random
from copy import copy
from dataclasses import dataclass
from datetime import datetime

from biocomp import datasets

OPERATORS = [
    '+',
    '-',
    '*',
    '/',
    '%',
]

EVALUATIONS = [
    '>',
    '<',
]


@dataclass
class CreationContext:
    min_depth: int
    max_depth: int
    feature_count: int
    min_const: int
    max_const: int
    mutation_rate: float
    crossover_rate: float

    def random_feature(self):
        return random.randint(0, self.feature_count - 1)

    def random_constant(self):
        return random.randint(self.min_const, self.max_const)

    def random_leaf(self):
        return random.choice((
            Tree('feature', self.random_feature(), None),
            Tree('constant', self.random_constant(), None),
        ))

    def grow(self, depth=0):
        depth += 1
        if depth < self.min_depth or depth < self.max_depth and random.choice((True, False)):
            return Tree(random.choice(OPERATORS), self.grow(depth), self.grow(depth))
        return self.random_leaf()

    def full(self, depth=0):
        depth += 1
        if depth < self.max_depth:
            return Tree(random.choice(OPERATORS), self.full(depth), self.full(depth))
        return self.random_leaf()


class Tree:
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, features):
        if not self.is_parent:
            if self.operator == 'feature':
                return features[self.left]
            if self.operator == 'constant':
                return self.left

        left = self.left.evaluate(features) if self.left is not None else 0
        right = self.right.evaluate(features) if self.left is not None else 0

        if self.operator == '+':
            return left + right

        if self.operator == '-':
            return left - right

        if self.operator == '*':
            return left * right

        if self.operator == '/':
            try:
                return left / right
            except ArithmeticError:
                return 1

        if self.operator == '%':
            try:
                return left % right
            except ArithmeticError:
                return 1

    def compress(self):
        if self.is_parent:
            if self.left.operator == 'constant' and self.right.operator == 'constant':
                return Tree('constant', self.evaluate([]), None)
            return Tree(self.operator, self.left.compress(), self.right.compress())
        return Tree(self.operator, self.left, self.right)

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    @property
    def is_parent(self):
        return isinstance(self.right, Tree) and isinstance(self.left, Tree)

    @property
    def depth(self):
        if self.is_parent:
            return max(self.right.depth, self.left.depth) + 1
        return 1

    def __copy__(self):
        return Tree(copy(self.operator), copy(self.left), copy(self.right))

    def __len__(self):
        if self.left is not None and self.right is not None:
            return len(self.left) + len(self.right) + 1
        return 1

    def __str__(self):
        if self.left is not None and self.right is not None:
            return f'({self.left} {self.operator} {self.right})'
        if self.operator == 'feature':
            return f'f{self.left}'
        return f'{self.left}'

    def __iter__(self):
        if isinstance(self.right, Tree):
            yield from iter(self.right)
        if isinstance(self.left, Tree):
            yield from iter(self.left)
        yield self


class Rule:
    def __init__(self, tree, evaluation, point):
        self.tree = tree
        self.evaluation = evaluation
        self.point = point

    @staticmethod
    def generate(context):
        tree = random.choice((context.grow, context.full))()
        evaluation = random.choice(EVALUATIONS)
        point = random.uniform(context.min_const, context.max_const)
        return Rule(tree, evaluation, point)

    def mutate(self, context):
        target = copy(self)

        if random.random() < context.mutation_rate:
            target.evaluation = random.choice(EVALUATIONS)

        if random.random() < context.mutation_rate:
            target.point = random.uniform(context.min_const, context.max_const)

        for subtree in iter(target.tree):
            if not subtree.is_parent:
                continue

            depth = subtree.depth

            if random.random() * (1 / depth) < context.mutation_rate:
                subtree.operator = random.choice(OPERATORS)

            if random.random() * (1 / depth) < context.mutation_rate:
                subtree.left = context.grow(target.tree.depth - subtree.depth + 1)

            if random.random() * (1 / depth) < context.mutation_rate:
                subtree.right = context.grow(target.tree.depth - subtree.depth + 1)

            if random.random() * (1 / depth) < context.mutation_rate:
                subtree.left, subtree.right = subtree.right, subtree.left

        return target

    def crossover(self, other, context):
        tree = copy(self.tree)

        if random.random() >= context.crossover_rate:
            return Rule(tree, self.evaluation, self.point)

        node_to_replace, replace_node_func = random.choice(list(itertools.chain(*(
            ((m.left, m.set_left), (m.right, m.set_right))
            for m in iter(tree)
            if m.is_parent
        ))))

        depth = tree.depth - node_to_replace.depth

        replacement = random.choice(list(itertools.chain(*(
            (n for n in (d.left, d.right) if context.min_depth <= n.depth + depth <= context.max_depth)
            for d in iter(copy(other.tree))
            if d.is_parent
        ))))

        replace_node_func(replacement)

        return Rule(tree, self.evaluation, self.point)

    def evaluate(self, features):
        value = self.tree.evaluate(features)

        if self.evaluation == '<':
            return value < self.point

        if self.evaluation == '>':
            return value > self.point

    def fitness(self, features, labels):
        correct = 0
        for f, l in zip(features, labels):
            if self.evaluate(f) == l:
                correct += 1
        generalisation = 1 / len(self.tree)
        return (correct + generalisation) / len(features)

    def compress(self):
        return Rule(self.tree.compress(), self.evaluation, self.point)

    def __copy__(self):
        return Rule(copy(self.tree), self.evaluation, self.point)

    def __str__(self):
        return f'{self.tree} {self.evaluation} {self.point}'


def select(population, fitnesses):
    winner, winner_fitness = None, -float('inf')
    for tournament in range(5):
        index = random.randint(0, len(population) - 1)
        if fitnesses[index] > winner_fitness:
            winner, winner_fitness = population[index], fitnesses[index]
    return winner


def run_experiment(context):
    features, labels, test_features, test_labels = datasets.split(datasets.load_dataset_2(), 0.8)
    population_size = 100
    population = [Rule.generate(context) for _ in range(population_size)]
    logfile = f'logs/{datetime.now()}.log'
    with open(logfile, 'a') as f:
        f.write(str(context) + '\n')

    for generation in range(500):
        fitnesses = [p.fitness(features, labels) for p in population]
        best_fitness = max(fitnesses)
        best = copy(population[fitnesses.index(best_fitness)])
        mean = sum(fitnesses) / len(fitnesses)

        test_fitnesses = [p.fitness(features, labels) for p in population]
        test_best_fitness = max(test_fitnesses)
        test_best = copy(population[fitnesses.index(test_best_fitness)])
        test_mean = sum(fitnesses) / len(fitnesses)

        if generation % 25 == 0:
            print(f'Generation: {generation:3} \t'
                  f'Best: {best_fitness:.3f} \t'
                  f'Mean: {mean:.3f} \t'
                  f'Depth: {best.tree.depth} \t'
                  f'Size: {len(best.tree)} \t'
                  f'Test Best: {test_best_fitness:.3f} \t'
                  f'Solution: {best}')

        with open(logfile, 'a') as f:
            time = str(datetime.now()).split('.')[0].replace(' ', '.')
            f.write(f'generation:{generation} '
                    f'time:{time} '
                    f'train_fitness_best:{best_fitness} '
                    f'train_fitness_mean:{mean} '
                    f'test_fitness_best:{test_best_fitness} '
                    f'test_fitness_mean:{test_mean}'
                    f'train_best:{str(best).replace(" ", "")} '
                    f'test_best:{str(test_best).replace(" ", "")}'
                    f'\n')

        population = [select(population, fitnesses) for _ in range(population_size)]
        population = [m.crossover(d, context) for m, d in zip(population, reversed(population))]
        population = [p.mutate(context) for p in population]
        population[-1] = best

        # compress
        for i in range(population_size):
            compressed = population[i].compress()
            if compressed.tree.depth >= context.min_depth:
                population[i] = compressed


def main():
    mutations = [0.0001, 0.001, 0.01, 0.05]
    crossovers = [0.1, 0.25, 0.5, 0.75, 0.9]

    for mutation_rate in mutations:
        for crossover_rate in crossovers:
            context = CreationContext(
                min_depth=4,
                max_depth=7,
                feature_count=6,
                min_const=0,
                max_const=2,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
            )

            for i in range(5):
                print(context)
                run_experiment(context)


if __name__ == '__main__':
    main()
