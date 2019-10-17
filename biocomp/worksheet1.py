import random

population_size = 50
gene_size = 64
generations = 1000
crossover_chance = 0.5
mutation_chance = 0.015
tournament_size = 5

population = [[random.choice([0, 1]) for _ in range(gene_size)] for _ in range(population_size)]

for generation in range(generations):
    population_fitness = [sum(gene) for gene in population]
    total_fitness = sum(population_fitness)
    best_fitness = max(population_fitness)
    best = population[population_fitness.index(best_fitness)].copy()
    print('best:', best_fitness, 'mean:', total_fitness / population_size)

    if best_fitness == gene_size:
        print('found solution at fitness', best_fitness, 'on generation', generation)
        break


    def select_parent():
        fittest, fitness = None, -1
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
