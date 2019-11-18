import os

import matplotlib.pyplot as plt

basedir = "c/logs/data1/"
with open(basedir + os.listdir(basedir)[5], "r") as f:
    lines = [l for l in f.readlines() if "generation" in l]
    datas = [dict(v.split(":", maxsplit=1) for v in l.split(" ")) for l in lines]

generations = list(range(len(datas)))

train_fitness_best = []
cross_validation_fitness_best = []
test_fitness_best = []

for i, d in enumerate(datas):
    train_fitness_best.append(float(d['train_fitness_best']))
    cross_validation_fitness_best.append(float(d['cross_validation_fitness_best']))
    test_fitness_best.append(float(d['test_fitness_best']))

plt.plot(
    generations, train_fitness_best,
    generations, cross_validation_fitness_best,
    generations, test_fitness_best,
    marker='o',
)
plt.show()
