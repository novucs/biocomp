import random

from biocomp import datasets


def dot(inputs, weights):
    step = len(weights) // len(inputs)
    return [sum(a * b for a, b in zip(inputs, weights[i * step:(i + 1) * step]))
            for i in range(step)]


def activate(x):
    return max(0, x)  # relu
    # return 1 / (1 + math.exp(-x))  # sigmoid


def rnn(size, inputs, weights):
    wx, wr = weights[:size], weights[size:]
    s = [0] * size
    for i in inputs:
        s = [activate(a + b) for a, b in zip(dot([i], wx), dot(s, wr))]
    return s


def net():
    rnn_size = 4
    rnn_weight_count = (1 * rnn_size) + (rnn_size ** 2)

    def predict(features, weights):
        a1 = rnn(rnn_size, features, weights[:rnn_weight_count])
        a2 = activate(dot(a1, weights[rnn_weight_count:])[0])
        prediction = 1 if a2 > 0.5 else 0
        # print(a2)
        return prediction

    total_weight_count = rnn_weight_count + rnn_size
    return predict, total_weight_count


def rand_weight():
    # return random.choice([-1., -.5, 0., .5, 1.])
    return random.uniform(-2, 2)


def main():
    features, labels, *_ = datasets.split(datasets.load_dataset_1())
    predict, gene_size = net()
    population_size = 50
    population = [[rand_weight() for _ in range(gene_size)]
                  for _ in range(population_size)]

    # population[0] = [0.5, 0.5, 1.0, 0.0, 1.0, -1.0, 0.0, -0.5, 0.0, 0.5, 1.0, 0.0, 0.5, -0.5, 0.5, -0.5, 0.0, -0.5, 1.0, 0.0, 1.0, 0.5, -0.5, 0.0]
    # population[1] = [0.5624719564907368, 0.01579871722570525, 0.8948303472886368, 0.6157340941722174, 0.9949089949082408, -0.9718687426685406, -0.04012044940035153, 1.929627787275046, 0.0, 1.5941283662940395, 0.9698990198034685, -0.5937242537595626, 0.5076766524913334, 0.31496156255221663, 0.57901642370433, -1.8955744904976095, -1.29001620980149, 1.8529476938214517, 1.534186975032509, 1.6210843111881692, 0.7812971429782949, 1.8966908800829234, 1.0429150609525122, 0.18841093752313087]

    generation_count = 10000
    tournament_size = 5
    crossover_chance = 0.25
    mutation_chance = 0.01
    best = []

    # # Attempt at brute force searching...
    # b, y = None, -float('inf')
    # import itertools
    # for p in itertools.product([-1.0, -0.5, 0.0, 0.5, 1.0], repeat=24):
    #     fns = sum(int(predict(f, p) == l) for f, l in zip(features, labels))
    #     if fns > y:
    #         b, y = p, fns
    #         print(y)

    for generation in range(generation_count):
        fitnesses = [sum(int(predict(f, g) == l) for f, l in zip(features, labels))
                     for g in population]
        best_fitness = max(fitnesses)
        best = population[fitnesses.index(best_fitness)].copy()
        print('Best:', best_fitness, 'Mean:', sum(fitnesses) / population_size)

        if best_fitness == len(features):
            print('Found solution in', generation, 'generations')
            print('Solution:', best)
            return

        def select():
            winner, winner_fitness = None, -float('inf')
            for _ in range(tournament_size):
                gene, fitness = random.choice(list(zip(population, fitnesses)))
                if fitness > winner_fitness:
                    winner, winner_fitness = gene, fitness
            return winner.copy()

        def crossover():
            if crossover_chance < random.random():
                return select()
            mother = select()
            father = select()
            index = int(random.random() * gene_size)
            return mother[:index] + father[index:]

        def mutate(gene):
            for index in range(len(gene)):
                if mutation_chance > random.random():
                    gene[index] = rand_weight()
            return gene

        population = [mutate(crossover()) for _ in range(population_size - 1)] + [best]

    print(best)


if __name__ == '__main__':
    main()
