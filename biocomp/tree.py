import copy
import itertools
import random
from typing import (
    Generator,
    List,
    Tuple,
    Union,
)

from biocomp import datasets


# formula:
# 2+(f1+f2)>threshold
#
# tree:
#    +
#   / \
#  2   +
#     / \
#    f1 f2

class Node:
    def evaluate(self, features):
        raise NotImplemented()


class LeafNode(Node):
    pass


class FeatureLeafNode(LeafNode):
    def __init__(self, feature):
        super(FeatureLeafNode, self).__init__()
        self.feature = feature

    def evaluate(self, features):
        return features[self.feature]

    def __str__(self):
        return f'f{self.feature}'


class ConstantLeafNode(LeafNode):
    def __init__(self, constant):
        super(ConstantLeafNode, self).__init__()
        self.constant = constant

    def evaluate(self, features):
        return self.constant

    def __str__(self):
        return str(self.constant)


class ZeroLeafNode(LeafNode):
    def __init__(self):
        super(ZeroLeafNode, self).__init__()
        self.constant = 0

    def evaluate(self, features):
        return self.constant

    def __str__(self):
        return str(self.constant)


class SubtreeNode(Node):
    def __init__(self, children=None):
        super(SubtreeNode, self).__init__()
        self.children: List[Node] = children or []


class AdditionSubtreeNode(SubtreeNode):
    def evaluate(self, features):
        return self.children[0].evaluate(features) + self.children[1].evaluate(features)

    def __str__(self):
        return f'({self.children[0]} + {self.children[1]})'


class ModuloSubtreeNode(SubtreeNode):
    def evaluate(self, features):
        if self.children[1] == 0 or self.children[1].evaluate(features) == 0:
            return 0
        return self.children[0].evaluate(features) % self.children[1].evaluate(features)

    def __str__(self):
        return f'({self.children[0]} % {self.children[1]})'


class DivisionSubtreeNode(SubtreeNode):
    def evaluate(self, features):
        if self.children[1] == 0:
            return 0
        return self.children[0].evaluate(features) / self.children[1].evaluate(features)

    def __str__(self):
        return f'({self.children[0]} / {self.children[1]})'


class MultiplicationSubtreeNode(SubtreeNode):
    def __init__(self, children=None):
        super().__init__(children)

    def evaluate(self, features):
        return self.children[0].evaluate(features) * self.children[1].evaluate(features)

    def __str__(self):
        return f'({self.children[0]} * {self.children[1]})'


class RootNode(SubtreeNode):
    def __init__(self, threshold, child, settings):
        super(RootNode, self).__init__()
        self.threshold: float = threshold
        self.children: List[Node] = [child]
        self.settings = settings

    def evaluate(self, features):
        return self.children[0].evaluate(features) < self.threshold

    def __str__(self):
        return f'{self.children[0]} < {self.threshold}'

    def fitness(self, features, labels):
        return sum(int(self.evaluate(f) == l)
                   for f, l in zip(features, labels)) / len(features)


def create_gene_inner(settings, depth=0) -> Node:
    depth += 1

    def create_node(node_create_settings):
        wheel = [settings[setting] for setting in node_create_settings.keys()]
        index = random.uniform(0, sum(wheel))
        cumulative = 0
        s = next(iter(node_create_settings.values()))
        for s, w in zip(node_create_settings.values(), wheel):
            cumulative += w
            if cumulative >= index:
                break
        return s()

    # Leaf creation
    def create_leaf_feature():
        feature = None
        while min(settings['running_feature_repeats']) < settings['max_feature_repeats']:
            feature = random.randint(0, settings['feature_count'] - 1)
            if settings['running_feature_repeats'][feature] < settings['max_feature_repeats']:
                break

        if feature is None:
            return None

        return FeatureLeafNode(feature)

    def create_leaf_constant():
        if settings['running_constant_repeats'] >= settings['max_constant_repeats']:
            return None

        value = random.uniform(settings['constant_min'], settings['constant_max'])
        return ConstantLeafNode(int(value) if settings['constant_int'] else value)

    def create_leaf():
        leaf = create_node({
            'odds_leaf_feature': create_leaf_feature,
            'odds_leaf_constant': create_leaf_constant,
        })

        return leaf if leaf is not None else ZeroLeafNode()

    # Subtree creation
    def create_children(count=2):
        return [create_gene_inner(settings, depth) for _ in range(count)]

    def create_subtree():
        return create_node({
            'odds_subtree_+': lambda: AdditionSubtreeNode(create_children()),
            'odds_subtree_%': lambda:  ModuloSubtreeNode(create_children()),
            'odds_subtree_/': lambda: DivisionSubtreeNode(create_children()),
            'odds_subtree_*': lambda: MultiplicationSubtreeNode(create_children()),
        })

    return create_leaf() if depth >= settings['max_depth'] else create_node({
        'odds_leaf': create_leaf,
        'odds_subtree': create_subtree,
    })


def create_gene_threshold(settings):
    return random.uniform(settings['threshold_min'], settings['threshold_max'])


def create_gene(settings) -> RootNode:
    settings = settings.copy()
    threshold = create_gene_threshold(settings)
    child = create_gene_inner(settings)
    return RootNode(threshold, child, settings)


def iterate_gene(gene, parent=None, path=None) -> \
        Generator[Tuple[Node, Union[RootNode, SubtreeNode], List[int]], None, None]:
    path = path or []

    if isinstance(gene, RootNode):
        yield from iterate_gene(gene.children[0], gene, path + [0])
        return

    if isinstance(gene, SubtreeNode):
        for index in range(len(gene.children)):
            yield from iterate_gene(gene.children[index], gene, path + [index])

    yield gene, parent, path


def crossover(mother: RootNode, father: RootNode, settings) -> \
        Tuple[RootNode, RootNode]:
    def random_node(gene: RootNode) -> Tuple[List[Node], int, int, int]:
        child, parent, path = random.choice(list(iterate_gene(gene)))
        size = max((len(path) for *_, path in iterate_gene(child)), default=0)
        return parent.children, path[-1], size, len(path)

    offspring_1 = copy.deepcopy(mother)
    offspring_2 = copy.deepcopy(father)

    root_1, index_1, size_1, depth_1 = random_node(offspring_1)
    root_2, index_2, size_2, depth_2 = random_node(offspring_2)
    max_depth = settings['max_depth']

    while (size_1 + depth_2) > max_depth or (size_2 + depth_1) > max_depth:
        root_1, index_1, size_1, depth_1 = random_node(offspring_1)
        root_2, index_2, size_2, depth_2 = random_node(offspring_2)

    root_1[index_1], root_2[index_2] = root_2[index_2], root_1[index_1]

    return offspring_1, offspring_2


def mutate(gene: RootNode, mutation_chance):
    gene = copy.deepcopy(gene)
    settings = gene.settings

    for child, parent, path in iterate_gene(gene):
        if random.uniform(0, 1) < mutation_chance:
            # update tree settings
            for grandchild, *_ in iterate_gene(parent.children[path[-1]]):
                if isinstance(grandchild, FeatureLeafNode):
                    settings['running_feature_repeats'][grandchild.feature] -= 1
                elif isinstance(grandchild, ConstantLeafNode):
                    settings['running_constant_repeats'] -= 1
            # replace child
            parent.children[path[-1]] = create_gene_inner(settings, len(path))
    if random.uniform(0, 1) < mutation_chance:
        gene.threshold = create_gene_threshold(settings)
    return gene


def tournament_selection(population, fitnesses, tournament_size):
    winner, winner_fitness = None, -float('inf')
    for tournament in range(tournament_size):
        index = random.randint(0, len(population) - 1)
        if fitnesses[index] > winner_fitness:
            winner, winner_fitness = population[index], fitnesses[index]
    return winner


def main():
    features, labels, *_ = datasets.split(datasets.load_dataset_2())
    population_size = 50
    crossover_chance = 0.0
    mutation_chance = 0.05
    tournament_size = 5
    tree_settings = {
        'max_depth': 7,
        'threshold_max': 1.0,
        'threshold_min': -1.0,
        'constant_max': 5.0,
        'constant_min': -5.0,
        'constant_int': True,
        'odds_subtree': 2.0,
        'odds_leaf': 1.0,
        'odds_leaf_feature': 1.0,
        'odds_leaf_constant': 0.1,
        'odds_leaf_features_sum': 0.0,
        'odds_subtree_+': 6.0,
        'odds_subtree_%': 1.0,
        'odds_subtree_/': 0.0,
        'odds_subtree_*': 0.0,
        'feature_count': len(features[0]),
        'max_feature_repeats': 1,
        'max_constant_repeats': 1,

        # running parameters
        'running_feature_repeats': [0] * len(features[0]),
        'running_constant_repeats': 0,
    }

    population = [create_gene(tree_settings) for _ in range(population_size)]
    overall_best, overall_best_fitness = None, -float('inf')

    generation = 0
    while True:
        generation += 1

        # evaluation
        fitnesses = [gene.fitness(features, labels) for gene in population]
        best_fitness = max(fitnesses)
        best = population[fitnesses.index(best_fitness)]
        mean = sum(fitnesses) / len(fitnesses)

        if best_fitness > overall_best_fitness:
            overall_best, overall_best_fitness = best, best_fitness
            with open('tree_solutions.txt', 'a') as f:
                print('New best:', overall_best_fitness, 'Gene:', overall_best)
                f.write(str(overall_best_fitness) + ' : ' + str(overall_best) + '\n')

            if best_fitness == 1:
                print('Model converged, best solution found on generation', generation)
                return

        print(generation, 'Best:', overall_best_fitness, 'Mean:', mean)

        # elitism
        population[fitnesses.index(min(fitnesses))] = copy.deepcopy(overall_best)

        # selection
        population = [
            tournament_selection(population, fitnesses, tournament_size)
            for _ in range(population_size)
        ]

        # crossover
        middle = population_size // 2
        population = list(itertools.chain(*(
            crossover(mother, father, tree_settings)
            if random.random() < crossover_chance else (mother, father)
            for mother, father in zip(population[:middle], population[middle:])
        )))

        # mutation
        population = [
            mutate(gene, mutation_chance)
            for gene in population
        ]


if __name__ == '__main__':
    main()
