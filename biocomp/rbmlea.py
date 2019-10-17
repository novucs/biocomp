import random

from biocomp import datasets

dataset = datasets.load_dataset_1()
train_x, train_y, *_ = datasets.split(dataset)

rule_count = 32
rule_size = len(train_x[0]) + 1
gene_size = rule_size * rule_count
population_size = 50
generation_count = 1000
crossover_chance = 0.5
mutation_chance = 0.0125
tournament_size = 5


def random_gene(index):
    return random.choice([0, 1, '#'] if index % rule_size == 0 else [0, 1])


population = [
    [random_gene(i) for i in range(rule_size * rule_count)]
    for _ in range(population_size)
]


def evaluate(gene, features):
    final_prediction = [0, 0]
    for index in range(0, gene_size, rule_size):
        *rule, prediction = gene[index: index + rule_size]
        if all(p == f or p == "#" for p, f in zip(rule, features)):
            final_prediction[prediction] += 1
    return final_prediction.index(max(final_prediction))


def fitness(gene, features, labels):
    return sum(1 if evaluate(gene, f) == l else 0
               for f, l in zip(features, labels)) / len(labels)


for generation in range(generation_count):
    population_fitness = [fitness(gene, train_x, train_y) for gene in population]
    best_fitness = max(population_fitness)
    best_gene_index = population_fitness.index(best_fitness)
    best_gene = population[best_gene_index]
    total_fitness = sum(population_fitness)
    mean_fitness = total_fitness / population_size

    if best_fitness == 1.0:
        print('found solution in', generation, 'generations')
        break

    print('best:', best_fitness, 'mean:', mean_fitness)


    def select_parent():
        fittest, fitness = None, -1
        for i in range(tournament_size):
            index = int(random.random() * population_size)
            if population_fitness[index] > fitness:
                fitness = population_fitness[index]
                fittest = population[index]
        return fittest


    def crossover():
        mother = select_parent()
        father = select_parent()
        return [random.choice((m, f)) for m, f in zip(mother, father)]


    population = [crossover() for _ in range(population_size - 1)]

    # mutate
    for gene in population:
        for index in range(gene_size):
            if mutation_chance > random.random():
                gene[index] = random_gene(index)

    population.append(best_gene)
