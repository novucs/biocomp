import math
import random

from biocomp import datasets


def dot(inputs, weights):
    step = len(weights) // len(inputs)
    return [sum(a * b for a, b in zip(inputs, weights[i * step:(i + 1) * step]))
            for i in range(step)]


def relu(x):
    return max(0, x)


def rnn(size, inputs, weights):
    wx, wr = weights[:size], weights[size:]
    s = [0] * size
    for i in inputs:
        s = [relu(a + b) for a, b in zip(dot([i], wx), dot(s, wr))]
    return s


def net():
    rnn_size = 4
    rnn_weight_count = (1 * rnn_size) + (rnn_size ** 2)

    def predict(features, weights):
        a1 = rnn(rnn_size, features, weights[:rnn_weight_count])
        a2 = relu(dot(a1, weights[rnn_weight_count:])[0])
        prediction = 1 if a2 > 0.5 else 0
        # print(a2)
        return prediction

    total_weight_count = rnn_weight_count + rnn_size
    return predict, total_weight_count


def main():
    features, labels, *_ = datasets.split(datasets.load_dataset_1())
    predict, gene_size = net()
    population_size = 50
    population = [[random.uniform(-1, 1) for _ in range(gene_size)]
                  for _ in range(population_size)]

    # for fid in range(32):
    #     print(features[fid], labels[fid], predict(features[fid], [
    #         0.7050795555114746, 1.3091033697128296, -0.8495133519172668,
    #           -1.4178032875061035,
    #         -1.380048394203186, 0.8350871205329895, 0.3204783499240875,
    #           -1.2147303819656372, 0.19507579505443573, 0.12077929824590683,
    #                                  0.2835042476654053, 0.5494593977928162,
    #              -1.48383367061615, 0.9164571762084961, 0.8283399343490601,
    #              -0.5954242944717407, 0.7464731335639954, 0.03303124010562897,
    #                                     1.4750480651855469, 1.8638049364089966,
    #         -0.7694922089576721, 0.5139452219009399, -0.7917553782463074,
    #             2.0158274173736572
    #
    #     ]))
    # return

    generation_count = 1000
    tournament_size = 5
    crossover_chance = 0.1
    mutation_chance = 0.00125

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
                    gene[index] = random.uniform(-1, 1)
            return gene

        population = [mutate(crossover()) for _ in range(population_size - 1)] + [best]


if __name__ == '__main__':
    main()
