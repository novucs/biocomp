import random

population_size = 100
gene_size = 10
generations = 1000
crossover_chance = 0.75
mutation_chance = 0.01

population = [[random.choice([0, 1]) for _ in range(gene_size)] for _ in range(population_size)]

for generation in range(generations):
    population_fitness = [sum(gene) for gene in population]
    total_fitness = sum(population_fitness)
    best_fitness = max(population_fitness)
    best = population[population_fitness.index(best_fitness)]
    print('best:', best_fitness, 'avg:', total_fitness / population_size)

    if best_fitness == gene_size:
        print('found solution', best, 'at fitness', best_fitness, 'on generation', generation)
        break


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
            return select_parent()
        mother = select_parent()
        father = select_parent()
        split = random.randint(0, len(mother) - 1)
        return mother[:split] + father[split:]


    population = [crossover() for _ in range(population_size - 1)]

    # mutation
    for gene in population:
        if mutation_chance < random.random():
            continue
        index = random.randint(0, gene_size - 1)
        gene[index] = int(not bool(gene[index]))

    # retention
    population.append(best)
