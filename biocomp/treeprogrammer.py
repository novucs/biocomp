from typing import List

from biocomp import datasets

dataset = datasets.load_dataset_1()
train_x, train_y, *_ = datasets.split(dataset)

gene_size = 5
population_size = 50
generation_count = 1000
crossover_chance = 0.5
mutation_chance = 0.0125
tournament_size = 5
