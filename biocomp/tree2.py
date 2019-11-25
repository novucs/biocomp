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


@dataclass
class CreationContext:
    depth: int
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
                return 0

        if self.operator == '%':
            try:
                return left % right
            except ArithmeticError:
                return 0

    def compress(self):
        if self.is_parent:
            if self.left.operator == 'constant' and self.right.operator == 'constant':
                return Tree('constant', self.evaluate([]), None)
            return Tree(self.operator, self.left.compress(), self.right.compress())
        return Tree(self.operator, self.left, self.right)

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


def grow(context):
    context.depth += 1

    if context.depth < context.min_depth or (context.depth < context.max_depth and random.choice((True, False))):
        operator = random.choice(OPERATORS)
        return Tree(operator, grow(copy(context)), grow(copy(context)))

    if random.choice((True, False)):
        return Tree('feature', context.random_feature(), None)

    return Tree('constant', context.random_constant(), None)


def mutate(tree, context):
    target = copy(tree)
    for subtree in iter(target):
        if not subtree.is_parent:
            if random.random() < context.mutation_rate:
                subtree.operator = random.choice(('feature', 'constant'))
                if subtree.operator == 'feature':
                    subtree.left = context.random_feature()
                if subtree.operator == 'constant':
                    subtree.left = context.random_constant()
            continue

        if random.random() < context.mutation_rate:
            subtree.operator = random.choice(OPERATORS)

        elif random.random() < context.mutation_rate:
            inner_context = copy(context)
            inner_context.depth = target.depth - subtree.depth + 1
            subtree.left = grow(inner_context)

        elif random.random() < context.mutation_rate:
            inner_context = copy(context)
            inner_context.depth = target.depth - subtree.depth + 1
            subtree.right = grow(inner_context)

    return target


def crossover(mum: Tree, dad: Tree, context: CreationContext):
    if random.random() >= context.crossover_rate:
        return copy(mum)

    dad = copy(dad)
    offspring = copy(mum)
    offspring_depth = offspring.depth
    inverse_size = 1 / (len(dad) + len(mum))

    for m in iter(offspring):
        if not m.is_parent:
            continue

        ml_depth = offspring_depth - m.left.depth
        mr_depth = offspring_depth - m.right.depth

        for d in iter(dad):
            if not d.is_parent:
                continue

            dl_depth = d.left.depth
            dr_depth = d.right.depth

            if dl_depth + ml_depth < context.max_depth and random.random() < inverse_size:
                m.left = d.left
                return offspring

            if dl_depth + mr_depth < context.max_depth and random.random() < inverse_size:
                m.right = d.left
                return offspring

            if dr_depth + ml_depth < context.max_depth and random.random() < inverse_size:
                m.left = d.right
                return offspring

            if dr_depth + mr_depth < context.max_depth and random.random() < inverse_size:
                m.right = d.right
                return offspring

    return offspring


def evaluate(root: Tree, features):
    return root.evaluate(features) < .5


def fitness(root: Tree, features, labels):
    correct = 0
    for f, l in zip(features, labels):
        if evaluate(root, f) == l:
            correct += 1
    generalisation = 1 / len(root)
    return (correct + generalisation) / len(features)


def select(population, fitnesses):
    winner, winner_fitness = None, -float('inf')
    for tournament in range(5):
        index = random.randint(0, len(population) - 1)
        if fitnesses[index] > winner_fitness:
            winner, winner_fitness = population[index], fitnesses[index]
    return winner


def main():
    features, labels, *_ = datasets.split(datasets.load_dataset_2())
    context = CreationContext(
        depth=0,
        min_depth=5,
        max_depth=16,
        feature_count=len(features[0]),
        min_const=0,
        max_const=2,
        mutation_rate=0.003,
        crossover_rate=0.85,
    )
    population_size = 100
    population = [grow(copy(context)) for _ in range(population_size)]
    logfile = f'logs/{datetime.now()}.log'

    for generation in range(5000):
        fitnesses = [fitness(p, features, labels) for p in population]
        best_fitness = max(fitnesses)
        best = copy(population[fitnesses.index(best_fitness)])
        mean = sum(fitnesses) / len(fitnesses)

        state = f'Generation: {generation:3} \tBest: {best_fitness:.3f} \tMean: {mean:.3f} \tDepth: {best.depth} \tSize: {len(best)} \tSolution: {best}'
        print(state)

        with open(logfile, 'a') as f:
            f.write(state + '\n')

        population = [select(population, fitnesses) for _ in range(population_size)]
        population = [crossover(m, d, copy(context)) for m, d in zip(population, reversed(population))]
        population = [mutate(p, copy(context)).compress() for p in population]
        population[-1] = best.compress()


if __name__ == '__main__':
    for i in range(30):
        main()
