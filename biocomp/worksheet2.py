import random

population_size = 50
gene_size = 64
generations = 10000
crossover_chance = 0.5
mutation_chance = 0.015
tournament_size = 5

population = [[random.choice([0, 1]) for _ in range(gene_size)] for _ in
              range(population_size)]


def binary_to_int(binary):
    sign, *binary = binary
    value = 0
    for bit in binary:
        value <<= 1
        value += bit
    return -value if sign else value


def gene_to_x_and_y(gene):
    index = len(gene) // 2
    correction = (binary_to_int([1] * index)) / 15
    x = binary_to_int(gene[:index]) / correction
    y = binary_to_int(gene[index:]) / correction
    return x, y


def fitness(gene):
    x, y = gene_to_x_and_y(gene)
    return 0 - (0.26 * (x ** 2 + y ** 2) - (0.48 * x * y))


for generation in range(generations):
    population_fitness = [fitness(gene) for gene in population]
    total_fitness = sum(population_fitness)
    best_fitness = max(population_fitness)
    best = population[population_fitness.index(best_fitness)].copy()
    print('best:', best_fitness, 'mean:', total_fitness / population_size)

    if best_fitness == 0:
        x, y = gene_to_x_and_y(best)
        print('found solution x:', x, 'y:', y,
              'at fitness', best_fitness,
              'on generation', generation)
        break


    def select_parent():
        fittest, fitness = None, -float('inf')
        for i in range(tournament_size):
            index = int(random.random() * population_size)
            if population_fitness[index] > fitness:
                fitness = population_fitness[index]
                fittest = population[index]
        return fittest


    def crossover():
        # if crossover_chance < random.uniform(0, 1):
        #     return select_parent()
        mother = select_parent()
        father = select_parent()
        return [random.choice((m, f)) for m, f in zip(mother, father)]


    population = [crossover() for _ in range(population_size - 1)]

    # mutation
    for gene in population:
        for index in range(gene_size):
            if mutation_chance > random.random():
                gene[index] = int(not bool(gene[index]))

    population.append(best)
