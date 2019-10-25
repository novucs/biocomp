import logging
import random

import xcs
from xcs.bitstrings import BitString
from xcs.scenarios import Scenario, ScenarioObserver

from biocomp import datasets


class HaystackProblem(Scenario):
    def __init__(self, training_cycles=100000, input_size=6):
        self.input_size = input_size
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        train_x, train_y, *_ = datasets.split(datasets.load_dataset_1())
        self.train_x = train_x
        self.train_y = train_y

    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.remaining_cycles = self.initial_training_cycles

    def more(self):
        return self.remaining_cycles > 0

    def sense(self):
        index = random.randint(0, len(self.train_x) - 1)
        haystack = BitString(self.train_x[index])
        self.needle_value = self.train_y[index]
        return haystack

    def execute(self, action):
        self.remaining_cycles -= 1
        return action == self.needle_value


# Setup logging so we can see the test run as it progresses.
logging.root.setLevel(logging.INFO)

# Create the scenario instance
problem = HaystackProblem()

# Wrap the scenario instance in an observer so progress gets logged,
# and pass it on to the test() function.
algorithm = xcs.XCSAlgorithm()

# # Default parameter settings in test()
# algorithm.exploration_probability = .5
#
# # Modified parameter settings
# algorithm.ga_threshold = 5
# algorithm.crossover_probability = .75
# algorithm.do_action_set_subsumption = True
# algorithm.do_ga_subsumption = True
# algorithm.wildcard_probability = .1
# algorithm.deletion_threshold = 20
# algorithm.mutation_probability = .002

xcs.test(algorithm, scenario=ScenarioObserver(problem))
