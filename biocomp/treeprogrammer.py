import copy
import random
from typing import List, Generator, Union, Tuple

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
#
# old representation:
# gene = (threshold, ('+', (2, ('+', ('f1', 'f2')))))
#
# new representation:
# gene = {
#   'type': 'root',
#   'threshold': 1,
#   'children': [
#       {
#           'type': 'addition',
#           'children': [
#               {
#                   'type': 'constant',
#                   'value': 2
#               },
#               {
#                   'type': 'addition',
#                   'children': [
#                       {
#                           'type': 'feature',
#                           'value': 1
#                       },
#                       {
#                           'type': 'feature',
#                           'value': 2
#                       }
#                   ]
#               }
#           ]
#       }
#   ]
# }

# def inner_evaluate_child(child, features) -> int:
#     if isinstance(child, int):
#         return child
#     if isinstance(child, str) and child.startswith('f'):
#         return features[int(child[1:])]
#     if isinstance(child, tuple):
#         return inner_evaluate(child, features)
#
#
# def inner_evaluate(gene, features):
#     operator, children = gene
#     children = [inner_evaluate_child(child, features) for child in children]
#
#     if operator == '+':
#         return sum(children)
#
#     if operator == '%':
#         return children[0] % children[1]
#
#     raise ValueError('Unknown operation:', operator)
#
#
# def evaluate(gene, features):
#     threshold, children = gene
#     result = inner_evaluate(children, features)
#     return 1 if result > threshold else 0


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
        if self.children[1] == 0:
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
    def evaluate(self, features):
        return self.children[0].evaluate(features) * self.children[1].evaluate(features)

    def __str__(self):
        return f'({self.children[0]} * {self.children[1]})'


class RootNode(SubtreeNode):
    def __init__(self, threshold, child):
        super(RootNode, self).__init__()
        self.threshold: float = threshold
        self.children: List[Node] = [child]

    def evaluate(self, features):
        return self.children[0].evaluate(features) < self.threshold

    def __str__(self):
        return f'{self.children[0]} < {self.threshold}'

    def fitness(self, features, labels):
        return sum(int(self.evaluate(f) == l)
                   for f, l in zip(features, labels)) / len(features)


def create_gene_inner(ls, gs) -> Node:
    ls = copy.deepcopy(ls)
    ls['depth'] += 1

    def create_node(settings):
        wheel = [gs[setting] for setting in settings.keys()]
        index = random.uniform(0, sum(wheel))
        cumulative = 0
        s = next(iter(settings.values()))
        for s, w in zip(settings.values(), wheel):
            cumulative += w
            if w >= index:
                break
        return s()

    # Leaf creation

    def create_leaf_feature():
        if gs['only_use_features_once']:
            if len(gs['unused_features']) <= 0:
                return create_leaf_constant()

            feature = random.choice(gs['unused_features'])
            gs['unused_features'].remove(feature)
        else:
            feature = random.randint(0, gs['feature_count'] - 1)
        return FeatureLeafNode(feature)

    def create_leaf_constant():
        return ConstantLeafNode(random.uniform(gs['constant_min'], gs['constant_max']))

    def create_leaf():
        return create_node({
            'odds_leaf_feature': create_leaf_feature,
            'odds_leaf_constant': create_leaf_constant,
        })

    # Subtree creation

    def create_children(count=2):
        return [create_gene_inner(ls, gs) for _ in range(count)]

    def create_subtree_addition():
        return AdditionSubtreeNode(create_children())

    def create_subtree_modulo():
        return ModuloSubtreeNode(create_children())

    def create_subtree_divide():
        return DivisionSubtreeNode(create_children())

    def create_subtree_multiply():
        return MultiplicationSubtreeNode(create_children())

    def create_subtree():
        return create_node({
            'odds_subtree_+': create_subtree_addition,
            'odds_subtree_%': create_subtree_modulo,
            'odds_subtree_/': create_subtree_divide,
            'odds_subtree_*': create_subtree_multiply,
        })

    return create_leaf() if ls['depth'] >= gs['max_depth'] else create_node({
        'odds_leaf': create_leaf,
        'odds_subtree': create_subtree,
    })


def create_gene_threshold():
    return random.uniform(-1.0, 1.0)


def create_gene_global_settings(feature_count):
    return {
        'feature_count': feature_count,
        'max_depth': 8,
        'only_use_features_once': True,
        'unused_features': list(range(feature_count)),
        'constant_max': 1.0,
        'constant_min': -1.0,
        'odds_subtree': 1.0,
        'odds_leaf': 1.0,
        'odds_leaf_feature': 8.0,
        'odds_leaf_constant': 1.0,
        'odds_leaf_features_sum': 1.0,
        'odds_subtree_+': 1.0,
        'odds_subtree_%': 1.0,
        'odds_subtree_/': 1.0,
        'odds_subtree_*': 1.0,
    }


def create_gene_local_settings():
    return {
        'depth': 0,
    }


def create_gene(feature_count) -> RootNode:
    global_settings = create_gene_global_settings(feature_count)
    local_settings = create_gene_local_settings()
    threshold = create_gene_threshold()
    child = create_gene_inner(local_settings, global_settings)
    return RootNode(threshold, child)


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


def crossover(mother: RootNode, father: RootNode, max_depth: int) -> \
        Tuple[RootNode, RootNode]:
    def random_node(gene: RootNode) -> Tuple[List[Node], int, int]:
        child, parent, path = random.choice(list(iterate_gene(gene)))
        size = max((len(path) for *_, path in iterate_gene(child)), default=0)
        return parent.children, path[-1], size

    offspring_1 = copy.deepcopy(mother)
    root_1, index_1, size_1 = random_node(offspring_1)

    offspring_2 = copy.deepcopy(father)
    root_2, index_2, size_2 = random_node(offspring_2)

    while (size_2 + size_1) > max_depth:
        root_1, index_1, size_1 = random_node(offspring_1)
        root_2, index_2, size_2 = random_node(offspring_2)

    root_1[index_1], root_2[index_2] = root_2[index_2], root_1[index_1]

    return offspring_1, offspring_2


def mutate(gene: RootNode, mutation_chance, feature_count):
    gene = copy.deepcopy(gene)
    for child, parent, path in iterate_gene(gene):
        if random.uniform(0, 1) < mutation_chance:
            global_settings = create_gene_global_settings(feature_count)
            local_settings = create_gene_local_settings()
            local_settings['depth'] = len(path)
            parent.children[path[-1]] = create_gene_inner(local_settings, global_settings)
    if random.uniform(0, 1) < mutation_chance:
        gene.threshold = create_gene_threshold()
    return gene


def tournament_selection(population, fitnesses, tournament_size):
    winner, winner_fitness = None, -float('inf')
    for tournament in range(tournament_size):
        index = random.randint(0, len(population) - 1)
        if fitnesses[index] > winner_fitness:
            winner, winner_fitness = population[index], fitnesses[index]
    return winner


def main():
    features, labels, *_ = datasets.split(datasets.load_dataset_1())
    population_size = 50
    generation_count = 10000
    crossover_chance = 0.85
    mutation_chance = 0.0125
    tournament_size = 5
    max_depth = 4

    population = [create_gene(len(features[0])) for _ in range(population_size)]
    overall_best, overall_best_fitness = None, -float('inf')

    # def predict(f0, f1, f2, f3, f4):
    #     return (((((((f1 + f2) + (f4 + f0)) % -0.4084233263502599) + f3) % 0.4290489863403546) % 0.5744925820063951) + 0.2948463671644945) > 0.5074617999721471
    # for f, l in zip(features, labels):
    #     print(f, predict(*f), l)
    # return

    for generation in range(generation_count):
        fitnesses = [gene.fitness(features, labels) for gene in population]
        best_fitness = max(fitnesses)
        best = population[fitnesses.index(best_fitness)]
        mean = sum(fitnesses) / len(fitnesses)

        if best_fitness > overall_best_fitness:
            overall_best, overall_best_fitness = best, best_fitness
            print('New best:', overall_best_fitness, 'Gene:', overall_best)

            if best_fitness == 1:
                print('Model converged, best solution found on generation', generation)
                return

        print(generation, 'Best:', overall_best_fitness, 'Mean:', mean)
        population[fitnesses.index(min(fitnesses))] = copy.deepcopy(overall_best)

        def select():
            return tournament_selection(population, fitnesses, tournament_size)

        # todo: tuple selection + crossover
        # crossover
        population = [
            crossover(select(), select(), max_depth)[0]
            if random.random() < crossover_chance else select()
            for _ in range(population_size)
        ]

        # mutation
        population = [
            mutate(gene, mutation_chance, len(features[0]))
            for gene in population
        ]

        # population.extend((create_gene(len(features[0])) for _ in range(50)))
        # population = [compress_gene(gene) for gene in population]


if __name__ == '__main__':
    main()
