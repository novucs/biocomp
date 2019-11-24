import random
from typing import (
    Generator,
    List,
    Tuple,
    Union,
)

from biocomp import datasets


class GlobalSettings:
    max_depth = 7
    max_size = 16
    threshold_max = 1.0
    threshold_min = -1.0
    constant_max = 5.0
    constant_min = -5.0
    constant_int = True
    odds_subtree = 2.0
    odds_leaf = 1.0
    odds_leaf_feature = 1.0
    odds_leaf_constant = 0.1
    odds_leaf_features_sum = 0.0
    odds_subtree_add = 6.0
    odds_subtree_mod = 1.0
    odds_subtree_mul = 0.0
    odds_subtree_div = 0.0
    max_feature_repeats = 1
    max_constant_count = 1
    feature_count = 6


class Node:
    def evaluate(self, features):
        raise NotImplemented()

    def copy(self):
        raise NotImplemented()


class LeafNode(Node):
    pass


class FeatureLeafNode(LeafNode):
    def __init__(self, feature):
        super(FeatureLeafNode, self).__init__()
        self.feature: int = feature

    def evaluate(self, features):
        return features[self.feature]

    def copy(self):
        return FeatureLeafNode(self.feature)

    def __str__(self):
        return f'f{self.feature}'


class ConstantLeafNode(LeafNode):
    def __init__(self, constant):
        super(ConstantLeafNode, self).__init__()
        self.constant: float = constant

    def evaluate(self, features):
        return self.constant

    def copy(self):
        return ConstantLeafNode(self.constant)

    def __str__(self):
        return str(self.constant)


class ZeroLeafNode(LeafNode):
    def __init__(self):
        super(ZeroLeafNode, self).__init__()
        self.constant = 0

    def evaluate(self, features):
        return self.constant

    def copy(self):
        return ZeroLeafNode()

    def __str__(self):
        return str(self.constant)


class SubtreeNode(Node):
    def __init__(self, children=None):
        super(SubtreeNode, self).__init__()
        self.children: List[Node] = children or []

    def copy(self):
        return self.__class__([child.copy() for child in self.children])


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
    def evaluate(self, features):
        return self.children[0].evaluate(features) * self.children[1].evaluate(features)

    def __str__(self):
        return f'({self.children[0]} * {self.children[1]})'


class RootNode(SubtreeNode):
    def __init__(self, threshold, child: Node, settings):
        super(RootNode, self).__init__()
        self.threshold: float = threshold
        self.children: List[Node] = [child]
        self.settings: NodeSettings = settings

    def evaluate(self, features):
        return self.children[0].evaluate(features) < self.threshold

    def copy(self):
        return RootNode(self.threshold, self.children[0].copy(), self.settings.copy())

    def __str__(self):
        return f'{self.children[0]} < {self.threshold}'

    def fitness(self, features, labels):
        max_size = GlobalSettings.max_depth ** 2
        size = node_size(self)

        generalisation = size / max_size
        correct_count = sum(int(self.evaluate(f) == l) for f, l in zip(features, labels))

        return (correct_count + generalisation) / len(features)

    def is_valid(self):
        if node_depth(self) > GlobalSettings.max_depth:
            return False

        if node_size(self) > self.settings.size:
            return False

        for repeats in self.settings.feature_repeats:
            if repeats > GlobalSettings.max_feature_repeats:
                return False

        if self.settings.constant_count > GlobalSettings.max_constant_count:
            return False

        if self.settings.constant_count != node_constant_count(self) or self.settings.feature_repeats != node_feature_repeats(self):
            return False

        return True


class NodeBuilder:
    def __init__(self, settings, depth=0):
        self.depth = depth
        self.settings: NodeSettings = settings

    def create_node(self, node_create_settings):
        index = random.uniform(0, sum(node_create_settings.keys()))
        cumulative = 0
        s = next(iter(node_create_settings.values()))
        for w, s in node_create_settings.items():
            cumulative += w
            if cumulative >= index:
                break
        return s()

    def create_leaf_feature(self):
        feature = None
        found = False
        while min(self.settings.feature_repeats) < GlobalSettings.max_feature_repeats:
            feature = random.randint(0, GlobalSettings.feature_count - 1)
            if self.settings.feature_repeats[feature] < GlobalSettings.max_feature_repeats:
                found = True
                break

        if not found:
            raise ValueError('overran on feature count')

        self.settings.feature_repeats[feature] += 1
        return FeatureLeafNode(feature)

    def create_leaf_constant(self):
        if self.settings.constant_count >= GlobalSettings.max_constant_count:
            raise ValueError('overran on constant count')

        self.settings.constant_count += 1
        value = random.uniform(GlobalSettings.constant_min, GlobalSettings.constant_max)
        return ConstantLeafNode(int(value) if GlobalSettings.constant_int else value)

    def create_leaf(self):
        allowed_creations = {}

        if min(self.settings.feature_repeats) < GlobalSettings.max_feature_repeats:
            allowed_creations[GlobalSettings.odds_leaf_feature] = self.create_leaf_feature

        if self.settings.constant_count < GlobalSettings.max_constant_count:
            allowed_creations[GlobalSettings.odds_leaf_constant] = self.create_leaf_constant

        leaf = self.create_node(allowed_creations)

        return leaf

    # Subtree creation
    def create_children(self):
        left = NodeBuilder(self.settings, self.depth + 1).build()
        right = NodeBuilder(self.settings, self.depth + 1).build()
        return [left, right]

    def create_subtree(self):
        return self.create_node({
            GlobalSettings.odds_subtree_add: lambda: AdditionSubtreeNode(self.create_children()),
            GlobalSettings.odds_subtree_mod: lambda: ModuloSubtreeNode(self.create_children()),
            GlobalSettings.odds_subtree_div: lambda: DivisionSubtreeNode(self.create_children()),
            GlobalSettings.odds_subtree_mul: lambda: MultiplicationSubtreeNode(self.create_children()),
        })

    def can_create_subtree(self):
        if self.depth >= GlobalSettings.max_depth:
            return False

        if self.settings.size >= GlobalSettings.max_size:
            return False

        expenses = sum(self.settings.feature_repeats) + self.settings.constant_count + self.depth + 1
        limits = (GlobalSettings.max_feature_repeats * len(
            self.settings.feature_repeats)) + GlobalSettings.max_constant_count

        if expenses >= limits:
            return False

        return True

    def build(self):
        self.settings.size += 1
        return self.create_leaf() if not self.can_create_subtree() else self.create_node({
            GlobalSettings.odds_leaf: self.create_leaf,
            GlobalSettings.odds_subtree: self.create_subtree,
        })


class NodeSettings:
    def __init__(self, feature_repeats, constant_count, size):
        self.feature_repeats = feature_repeats
        self.constant_count = constant_count
        self.size = size

    def copy(self):
        return NodeSettings(self.feature_repeats, self.constant_count, self.size)


def create_threshold():
    return random.uniform(GlobalSettings.threshold_min, GlobalSettings.threshold_max)


def create_node() -> RootNode:
    settings = NodeSettings(
        feature_repeats=[0] * GlobalSettings.feature_count,
        constant_count=0,
        size=1,
    )
    threshold = create_threshold()
    child = NodeBuilder(settings).build()
    return RootNode(threshold, child, settings)


def iterate_node(node, parent=None, path=None) -> \
        Generator[Tuple[Node, Union[RootNode, SubtreeNode], List[int]], None, None]:
    path = path or []

    if isinstance(node, RootNode):
        yield from iterate_node(node.children[0], node, path + [0])
        return

    if isinstance(node, SubtreeNode):
        for index in range(len(node.children)):
            yield from iterate_node(node.children[index], node, path + [index])

    yield node, parent, path


def node_depth(node: Node):
    return max((len(path) for *_, path in iterate_node(node)), default=0)


def node_size(node: Node):
    return len(list(iterate_node(node)))


def random_node(node: Node) -> Tuple[Node, Union[RootNode, SubtreeNode], List[int]]:
    return random.choice(list(iterate_node(node)))


def node_feature_repeats(node: Node):
    repeats = [0] * GlobalSettings.feature_count
    for child, parent, path in iterate_node(node):
        if isinstance(child, FeatureLeafNode):
            repeats[child.feature] += 1
    return repeats


def node_constant_count(node: Node):
    count = 0
    for child, parent, path in iterate_node(node):
        if isinstance(child, ConstantLeafNode):
            count += 1
    return count


def crossover(mum: RootNode, dad: RootNode) -> RootNode:
    for i in range(10):
        offspring = mum.copy()
        child, parent, path = random_node(offspring)
        replacement, *_ = random_node(dad.copy())
        parent.children[path[-1]] = replacement
        offspring.settings.feature_repeats = node_feature_repeats(offspring)
        offspring.settings.constant_count = node_constant_count(offspring)
        offspring.settings.size = node_size(offspring)

        if offspring.is_valid():
            return offspring

    return mum.copy()


def mutate(node: RootNode, mutation_chance):
    original = node.copy()

    for i in range(100):
        node = original.copy()
        settings = node.settings

        for child, parent, path in iterate_node(node):
            if random.uniform(0, 1) < mutation_chance:
                # update tree settings
                for grandchild, *_ in iterate_node(parent.children[path[-1]]):
                    settings.size -= 1
                    if isinstance(grandchild, FeatureLeafNode):
                        settings.feature_repeats[grandchild.feature] -= 1
                    elif isinstance(grandchild, ConstantLeafNode):
                        settings.constant_count -= 1
                # replace child
                replacement = NodeBuilder(settings, len(path)).build()
                parent.children[path[-1]] = replacement
        if random.uniform(0, 1) < mutation_chance:
            node.threshold = create_threshold()

        if node.is_valid():
            return node
    print('help')
    return None


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
    crossover_chance = 0.85
    mutation_chance = 0.003
    tournament_size = 5
    GlobalSettings.feature_count = len(features[0])
    population = [create_node() for _ in range(population_size)]
    overall_best, overall_best_fitness = None, -float('inf')

    generation = 0
    while True:
        generation += 1

        # evaluation
        fitnesses = [individual.fitness(features, labels) for individual in population]
        best_fitness = max(fitnesses)
        best = population[fitnesses.index(best_fitness)]
        mean = sum(fitnesses) / len(fitnesses)

        if best_fitness > overall_best_fitness:
            overall_best, overall_best_fitness = best, best_fitness
            with open('tree_solutions.txt', 'a') as f:
                print('New best:', overall_best_fitness, 'Solution:', overall_best)
                f.write(str(overall_best_fitness) + ' : ' + str(overall_best) + '\n')

            if best_fitness == 1:
                print('Model converged, best solution found on generation', generation)
                return

        print(generation, 'Best:', overall_best_fitness, 'Mean:', mean)

        # elitism
        population[-1] = overall_best.copy()

        # selection
        population = [
            tournament_selection(population, fitnesses, tournament_size)
            for _ in range(population_size)
        ]

        # crossover
        population = [
            crossover(mum, dad) if random.random() < crossover_chance
            else mum
            for mum, dad in zip(population, reversed(population))
        ]

        if any(not p.is_valid() for p in population):
            print('x')

        # mutation
        population = [mutate(individual, mutation_chance) for individual in population]

        if any(not p.is_valid() for p in population):
            print('x')

if __name__ == '__main__':
    main()
