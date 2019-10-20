import copy
import random

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


def create_gene_inner(ls, gs):
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
        return {
            'type': 'feature',
            'value': feature,
            'evaluate': lambda f: f[feature],
            'pretty': lambda: f'f{feature}',
        }

    def create_leaf_constant():
        value = random.uniform(gs['constant_min'], gs['constant_max'])
        return {
            'type': 'constant',
            'value': value,
            'evaluate': lambda f: value,
            'pretty': lambda: str(value),
        }

    def create_leaf():
        return create_node({
            'odds_leaf_feature': create_leaf_feature,
            'odds_leaf_constant': create_leaf_constant,
        })

    # Subtree creation

    def create_children(count=2):
        return [create_gene_inner(ls, gs) for _ in range(count)]

    def create_subtree_addition():
        children = create_children()
        return {
            'type': 'addition',
            'children': children,
            'evaluate': lambda f: sum(child['evaluate'](f) for child in children),
            'pretty': lambda: f"({children[0]['pretty']()} + {children[1]['pretty']()})",
        }

    def create_subtree_modulo():
        children = create_children()

        def evaluate_modulo(f):
            a, b = children[0]['evaluate'](f), children[1]['evaluate'](f)
            if b == 0:
                return 0
            return a % b

        return {
            'type': 'modulo',
            'children': children,
            'evaluate': evaluate_modulo,
            'pretty': lambda: f"({children[0]['pretty']()} % {children[1]['pretty']()})",
        }

    def create_subtree():
        return create_node({
            'odds_subtree_+': create_subtree_addition,
            'odds_subtree_%': create_subtree_modulo,
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
        'max_depth': 12,
        'only_use_features_once': True,
        'unused_features': list(range(feature_count)),
        'constant_max': 1.0,
        'constant_min': -1.0,
        'odds_subtree': 1.0,
        'odds_leaf': 1.0,
        'odds_leaf_feature': 1.0,
        'odds_leaf_constant': 1.0,
        'odds_subtree_+': 1.0,
        'odds_subtree_%': 1.0,
    }


def create_gene_local_settings():
    return {
        'depth': 0,
    }


def create_gene(feature_count):
    global_settings = create_gene_global_settings(feature_count)
    local_settings = create_gene_local_settings()
    child = create_gene_inner(local_settings, global_settings)
    threshold = create_gene_threshold()
    return {
        'type': 'root',
        'threshold': threshold,
        'children': [child],
        'evaluate': lambda f: 1 if child['evaluate'](f) > threshold else 0,
        'pretty': lambda: f"{child['pretty']()} < {threshold}",
    }


def iterate_gene_child_view(gene, path=None):
    path = path or []

    if 'children' not in gene:
        yield path, gene
        return

    for index, child in enumerate(gene['children']):
        child_path = path + [index]
        yield from iterate_gene_child_view(child, child_path)

    yield path, gene
    return


def iterate_gene_parent_view(gene, depth=0):
    if 'children' not in gene:
        return

    for index, child in enumerate(gene['children']):
        yield from iterate_gene_parent_view(child, depth + 1)
        yield gene, index, depth


def evaluate(gene, features):
    return gene['evaluate'](features)


def fitness(gene, features, labels):
    return sum(int(evaluate(gene, f) == l) for f, l in zip(features, labels))


def pretty(gene):
    return gene['pretty']()


def crossover(mother, father, max_depth):
    def random_node(gene):
        perm = list(iterate_gene_parent_view(gene))
        index = random.randint(0, len(perm) - 1)
        parent, child_index, depth = perm[index]
        child = parent['children'][child_index]
        size = max((depth for *_, depth in iterate_gene_parent_view(child)), default=0)
        return parent['children'], child_index, size

    offspring_1 = copy.deepcopy(mother)
    root_1, index_1, size_1 = random_node(offspring_1)

    offspring_2 = copy.deepcopy(father)
    root_2, index_2, size_2 = random_node(offspring_2)

    while (size_2 + size_1) > max_depth:
        root_1, index_1, size_1 = random_node(offspring_1)
        root_2, index_2, size_2 = random_node(offspring_2)

    root_1[index_1], root_2[index_2] = root_2[index_2], root_1[index_1]

    return offspring_1, offspring_2


def mutate(gene, mutation_chance, feature_count):
    gene = copy.deepcopy(gene)
    for node, index, depth in iterate_gene_parent_view(gene):
        if random.uniform(0, 1) < mutation_chance:
            global_settings = create_gene_global_settings(feature_count)
            local_settings = create_gene_local_settings()
            local_settings['depth'] = depth
            node['children'][index] = create_gene_inner(local_settings, global_settings)
    if random.uniform(0, 1) < mutation_chance:
        gene['threshold'] = create_gene_threshold()
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
    crossover_chance = 0.5
    mutation_chance = 0.1
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
        fitnesses = [fitness(gene, features, labels) for gene in population]
        best_fitness = max(fitnesses)
        best = population[fitnesses.index(best_fitness)]
        mean = sum(fitnesses) / population_size

        if best_fitness > overall_best_fitness:
            overall_best, overall_best_fitness = best, best_fitness
            print('New best:', overall_best_fitness, 'Gene:', pretty(overall_best))

            if best_fitness == len(features):
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
            mutate(gene, mutation_chance, len(features))
            if random.random() < mutation_chance else gene
            for gene in population
        ]

        population.extend((create_gene(len(features[0])) for _ in range(50)))


if __name__ == '__main__':
    main()
