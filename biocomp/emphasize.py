import random
from typing import (
    Dict,
    Optional,
    Union,
)

from biocomp import datasets

train_x, train_y, *_ = datasets.split(datasets.load_dataset_2())
population_count = 50
rule_size = len(train_x[0]) + 1
rule_count_per_chromosome = 17
chromosome_size = rule_size * rule_count_per_chromosome
generation_count = 10000
tournament_size = 5
crossover_chance = 0.85
mutation_chance = 0.5 / chromosome_size

mutation_difference_max_threshold = 0.10
mutation_difference_min_threshold = 0.05
mutation_update_size = mutation_chance / 100


def gene_by_index(index, emphasize_settings: Optional[Dict[Union[int, str], int]] = None):
    if emphasize_settings is None:
        return random.choice(
            [0, 1, '#'] if index % rule_size != (rule_size - 1)
            else [0, 1]
        )

    emphasize_settings = emphasize_settings.copy()
    if index % rule_size == (rule_size - 1):
        del emphasize_settings['#']

    total = sum(emphasize_settings.values())
    point = random.randint(0, total - 1)
    cumulative = 0

    for key, value in emphasize_settings.items():
        cumulative += value
        if cumulative > point:
            return key


def create_chromosome():
    return [gene_by_index(index) for index in range(chromosome_size)]


population = [create_chromosome() for _ in range(population_count)]


def evaluate(chromosome, features):
    prediction = [0, 0]
    for index in range(0, chromosome_size, rule_size):
        *condition, label = chromosome[index:index + rule_size]
        if all(p == f or p == '#' for p, f in zip(condition, features)):
            prediction[label] += 1
    return prediction.index(max(prediction))


def fitness(chromosome):
    return sum(int(evaluate(chromosome, features) == label)
               for features, label in zip(train_x, train_y)) / len(train_x)


overall_best, overall_best_fitness = None, -float('inf')

for generation in range(generation_count):
    fitnesses = [fitness(chromosome) for chromosome in population]
    best_fitness = max(fitnesses)
    best = population[fitnesses.index(best_fitness)]
    mean = sum(fitnesses) / len(population)

    if generation % 5 == 0:
        print('generation:', generation, 'Best:', best_fitness, 'Mean:', mean, 'Rate:', mutation_chance)

    # if best_fitness - mean > mutation_difference_max_threshold:
    #     mutation_chance -= mutation_update_size
    # if best_fitness - mean < mutation_difference_min_threshold:
    #     mutation_chance += mutation_update_size

    if overall_best_fitness < best_fitness:
        print('New best fitness:', best_fitness, 'Solution:', best)
        overall_best, overall_best_fitness = best, best_fitness
        if best_fitness == 1.0:
            print('Found solution on generation', generation)
            break

    population_batch_size = population_count // 5
    sorted_population = sorted(zip(population, fitnesses), key=lambda a: a[1])
    best_batch = sorted_population[:population_batch_size]
    random_batch = random.choices(sorted_population, k=population_batch_size)
    parent_batch = best_batch + random_batch

    def select_parent():
        parent, parent_fitness = None, -1
        for i in range(tournament_size):
            candidate, candidate_fitness = random.choice(sorted_population)
            if candidate_fitness > parent_fitness:
                parent, parent_fitness = candidate, candidate_fitness
        return parent.copy()

    def crossover():
        mother = select_parent()
        father = select_parent()
        return [random.choice((m, f)) for m, f in zip(mother, father)]

    # population = [crossover() for _ in range(population_batch_size - 1)] \
    #     + [chromosome for chromosome, fitness in best_batch] \
    #     + [chromosome for chromosome, fitness in random_batch]
    population = [crossover() for _ in range(population_count)]

    # mutate
    emphasize_modifiers = {gene: best.count(gene) for gene in [0, 1, '#']}
    for chromosome in population:
        for index in range(chromosome_size):
            if mutation_chance > random.random():
                chromosome[index] = gene_by_index(index, {
                    gene: (2 if best[index] == gene else 1) * emphasize_modifiers[gene]
                    for gene in [0, 1, '#']
                })

    population.append(overall_best.copy())
