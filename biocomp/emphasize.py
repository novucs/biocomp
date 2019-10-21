import random
from typing import (
    Dict,
    Optional,
    Union,
)

from biocomp import datasets

train_x, train_y, *_ = datasets.split(datasets.load_old_dataset_1())
population_count = 50
rule_size = len(train_x[0]) + 1
rule_count_per_gene = 17
gene_size = rule_size * rule_count_per_gene
generation_count = 10000
tournament_size = 5
crossover_chance = 0.85
mutation_chance = 0.0125


def gene_at(index, settings: Optional[Dict[Union[int, str], int]] = None):
    if settings is None:
        return random.choice(
            [0, 1, '#'] if index % rule_size != (rule_size - 1)
            else [0, 1]
        )

    settings = settings.copy()
    if index % rule_size == (rule_size - 1):
        del settings['#']

    total = sum(settings.values())
    point = random.randint(0, total - 1)
    cumulative = 0

    for key, value in settings.items():
        cumulative += value
        if cumulative > point:
            return key


def create_gene():
    return [gene_at(index) for index in range(gene_size)]


population = [create_gene() for _ in range(population_count)]


def evaluate(gene, features):
    prediction = [0, 0]
    for index in range(0, gene_size, rule_size):
        *condition, label = gene[index:index + rule_size]
        if all(p == f or p == '#' for p, f in zip(condition, features)):
            prediction[label] += 1
    return prediction.index(max(prediction))


def fitness(gene):
    return sum(int(evaluate(gene, features) == label)
               for features, label in zip(train_x, train_y)) / len(train_x)


overall_best, overall_best_fitness = None, -float('inf')

for generation in range(generation_count):
    fitnesses = [fitness(gene) for gene in population]
    best_fitness = max(fitnesses)
    best = population[fitnesses.index(best_fitness)]
    mean = sum(fitnesses) / len(population)

    if generation % 5 == 0:
        print('Generation:', generation, 'Best:', best_fitness, 'Mean:', mean)

    if overall_best_fitness < best_fitness:
        print('New best fitness:', best_fitness, 'Solution:', best)
        overall_best, overall_best_fitness = best, best_fitness
        if best_fitness == 1.0:
            print('Found solution on generation', generation)
            break


    def select_parent():
        fittest, fitness = None, -1
        for i in range(tournament_size):
            index = int(random.random() * population_count)
            if fitnesses[index] > fitness:
                fitness = fitnesses[index]
                fittest = population[index]
        return fittest.copy()


    def crossover():
        mother = select_parent()
        father = select_parent()
        return [random.choice((m, f)) for m, f in zip(mother, father)]


    population = [
        crossover() if crossover_chance > random.random() else select_parent()
        for _ in range(population_count - 1)
    ]

    mutation_settings = {
        cell: best.count(cell)
        for cell in [0, 1, '#']
    }

    # mutate
    for gene in population:
        for index in range(gene_size):
            if mutation_chance > random.random():
                # standard
                # gene[index] = gene_at(index)

                # aggregate
                gene[index] = gene_at(index, mutation_settings)

                # positional
                # gene[index] = gene_at(index, {
                #     cell: 2 if best[index] == cell else 1
                #     for cell in [0, 1, '#']
                # })

    population.append(best)
