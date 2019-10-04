import itertools
import random

from biocomp import datasets

dataset = datasets.load_dataset_2()
train_x, train_y, *_ = datasets.split(dataset, 0.9)

gene = [0, 1]
condition_count = 5
condition_size = len(train_x[0])
gene_size = condition_size * condition_count + 1
population_size = 100
population = [
    [random.choice([0, 1, "#"])
     for _ in range(condition_size * condition_count)] + [random.choice([0, 1])]
    for _ in range(population_size)
]
generation_count = 1000
crossover_chance = 0.75
mutation_chance = 0.05


def evaluate(gene, features):
    prediction = gene[-1]
    for index in range(0, gene_size - 1, condition_size):
        condition = gene[index: index + condition_size]
        condition_success = all(p == f or p == "#" for p, f in zip(condition, features))
        if not condition_success:
            return 0 if prediction else 1
    return prediction


def fitness(gene, features, labels):
    return sum(1 if evaluate(gene, f) == l else 0 for f, l in zip(features, labels)) / len(labels)


def mutate(gene):
    if random.uniform(0, 1) > mutation_chance:
        return gene
    index = random.randint(0, gene_size - 1)
    if index == (gene_size - 1):
        replacement = random.choice([0, 1])
    else:
        replacement = random.choice([0, 1, '#'])
    gene[index] = replacement
    return gene


for generation in range(generation_count):
    population_fitness = [fitness(gene, train_x, train_y) for gene in population]
    best_gene_index = population_fitness.index(max(population_fitness))
    best_gene = population[best_gene_index]
    total_fitness = sum(population_fitness)
    mean_fitness = total_fitness / population_size

    print('best:', population_fitness[best_gene_index], 'mean:', mean_fitness)


    def select_parent():
        target = random.uniform(0, total_fitness)
        partial = 0
        for chromosome, fitness in zip(population, population_fitness):
            partial += fitness
            if partial > target:
                return chromosome
        return random.choice(population)


    def crossover():
        if crossover_chance < random.uniform(0, 1):
            return select_parent().copy()

        mother = select_parent()
        father = select_parent()
        index = random.randint(0, gene_size - 1)
        return mother[:index] + father[index:]


    population = [crossover() for _ in range(population_size - 1)]
    population = [mutate(gene) for gene in population]
    population.append(best_gene)
