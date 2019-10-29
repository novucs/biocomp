import random

from biocomp import datasets

dataset = datasets.load_dataset_2()
train_x, train_y, *_ = datasets.split(dataset)

rule_count = 30
rule_size = len(train_x[0]) + 1
chromosome_size = rule_size * rule_count
population_size = 50
generation_count = 10000
crossover_chance = 0.5
mutation_chance = 0.0125
tournament_size = 5


def random_chromosome(index):
    return random.choice([0, 1, '#'] if index % rule_size != (rule_size - 1) else [0, 1])
    # return random.choice([0, 1])


population = [
    [random_chromosome(i) for i in range(rule_size * rule_count)]
    for _ in range(population_size)
]


def evaluate(chromosome, features):
    final_prediction = [0, 0]
    for index in range(0, chromosome_size, rule_size):
        *rule, prediction = chromosome[index: index + rule_size]
        if all(p == f or p == "#" for p, f in zip(rule, features)):
            return prediction
    # print(final_prediction)
    return 0
    # return 0


def fitness(chromosome, features, labels):
    # for f, l in zip(features, labels):
    #     prediction = bool(evaluate(chromosome, f))
    #     print(f, l, prediction, '<---' if prediction != l else '')
    return sum(1 if evaluate(chromosome, f) == l else 0
               for f, l in zip(features, labels)) / len(labels)


overall_best, overall_best_fitness = None, -float('inf')

for generation in range(generation_count):
    # population = [[
    #     0, 0,  '#', '#', '#', 1, 1,
    #     1, 1, 1,  '#', '#', '#', 1,
    #     0, 1,  '#', '#', 1, '#', 1,
    #     1, 0,  '#', 1, '#', '#', 1,
    #     '#', '#', '#', '#', '#', '#', 0,
    # ]]
    # population = [[1, 1, 1, '#', '#', '#', 1, 0, 0, '#', '#', '#', 1, 1, 0, 1, 0, '#', 1, '#', 1, 1, 0, '#', 1, '#', '#', 1, '#', 1, 1, '#', 1, '#', 1]]
    population_fitness = [fitness(chromosome, train_x, train_y) for chromosome in population]
    # break
    best_fitness = max(population_fitness)
    best_index = population_fitness.index(best_fitness)
    best = population[best_index]
    total_fitness = sum(population_fitness)
    mean_fitness = total_fitness / population_size

    if best_fitness > overall_best_fitness:
        print('new overall best:', best)
        overall_best, overall_best_fitness = best, best_fitness

        if best_fitness == 1.0:
            print('Found solution in', generation, 'generations:')
            for rule_id, index in enumerate(range(0, chromosome_size, rule_size)):
                *rule, prediction = best[index: index + rule_size]
                print(f'\tRule #{rule_id + 1}', rule, ':', prediction)
            break

    print('generation', generation, 'best:', best_fitness, 'mean:', mean_fitness)


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
    for chromosome in population:
        for index in range(chromosome_size):
            if mutation_chance > random.random() and index % rule_size != (rule_size - 1):
                chromosome[index] = random_chromosome(index)

    population.append(best)
