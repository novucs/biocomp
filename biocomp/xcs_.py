# Implementation of XCS as is defined in the paper:
# "An Algorithmic Description of XCS" by Martin V. Butz and Stewart W. Wilson.
import copy
import random

from biocomp import datasets

train_x, train_y, test_x, test_y = datasets.split(datasets.load_dataset_1(), 0.9)
action_space = set(train_y)

# XCS hyperparameters
condition_size = len(train_x[0])  # L
max_population_size = 200  # N
learning_rate = 0.15  # β
accuracy_coefficient = 0.1  # α
# discount factor, only for multi-step problems
discount_factor = 0.71  # γ
# specifies frequency of GA executions (by average last ga usage in population)
# often between 25-50
ga_threshold = 35  # θGA
crossover_probability = 0.75  # χ
mutation_probability = 0.03  # μ
# if classifier experience > threshold, its fitness may be considered in the
# deletion process
deletion_threshold = 20  # θdel
fitness_threshold = 0.1  # δ
subsumption_threshold = 20  # θsub
# probability of using a # attribute during covering
wildcard_probability = 0.33  # P#
initial_prediction = 0.00001  # pI
initial_error = 0.00001  # εI
initial_fitness = 0.00001  # fI
exploration_probability = 0.5  # pexplr
covering_threshold = len(action_space)  # θmna
do_ga_subsumption = False
do_action_set_subsumption = False

# reinforcement program hyperparameters
maximum_reward = 1000
minimum_reward = 0

# error below which classifiers are considered to have equal accuracy
# ε0 = 1% maximum payoff
error_clip = maximum_reward / 100  # ε0


class Classifier:  # cl
    def __init__(self, condition, action, created):
        self.condition = condition  # C
        self.action = action  # A
        self.payoff = initial_prediction  # p
        self.error = initial_error  # ε
        self.fitness = initial_fitness  # f
        # number of times this classifier has been in an action set
        self.experience = 0.0  # exp
        # timestamp for when this classifier was last in the action set while
        # the genetic algorithm was being executed
        self.last_in_ga = created  # ts
        # the average size of the action sets this classifier has belonged to
        self.niche = 1.0  # as
        # number of macro classifiers this classifier represents
        self.numerosity = 1  # num

    def matches(self, attributes):
        return all(c == a or c == '#' for c, a in zip(self.condition, attributes))

    def could_subsume(self):
        if self.experience > subsumption_threshold:
            if self.error < error_clip:
                return True
        return False

    def is_more_general_than(self, other):
        if self.condition.count('#') <= other.condition.count('#'):
            return False
        for attribute_a, attribute_b in zip(self.condition, other.condition):
            if attribute_a != '#' and attribute_a != attribute_b:
                return False
        return True

    def does_subsume(self, other):
        return self.action == other.action and self.could_subsume() \
               and self.is_more_general_than(other)


class XCS:
    def __init__(self):
        self.population = []  # [P]
        self.actual_time = 0  # t

    def deletion_vote(self, classifier, mean_fitness):
        vote = classifier.niche * classifier.numerosity
        single_fitness = classifier.fitness / classifier.numerosity
        if classifier.experience > deletion_threshold \
                and single_fitness < fitness_threshold * mean_fitness:
            vote *= mean_fitness / single_fitness
        return vote

    def delete_from_population(self, population):
        population_size = sum(c.numerosity for c in population)
        if population_size <= max_population_size:
            return
        mean_fitness = sum(c.fitness for c in population) / population_size
        vote_sum = sum(
            self.deletion_vote(classifier, mean_fitness)
            for classifier in population
        )
        choice_point = random.random() * vote_sum
        counter = 0
        for classifier in population:
            counter += self.deletion_vote(classifier, mean_fitness)
            if counter > choice_point:
                if classifier.numerosity > 1:
                    classifier.numerosity -= 1
                else:
                    population.remove(classifier)
                return

    def generate_covering_classifier(self, match_set, attributes):
        condition = [
            '#' if random.random() < wildcard_probability else attribute
            for attribute in attributes
        ]
        action = random.choice(list(action_space - {c.action for c in match_set}))
        return Classifier(condition, action, self.actual_time)

    def generate_match_set(self, attributes):
        while True:
            match_set = {
                classifier
                for classifier in self.population
                if classifier.matches(attributes)
            }
            action_count = len({c.action for c in match_set})
            if action_count < covering_threshold:
                classifier = self.generate_covering_classifier(match_set, attributes)  # clc
                self.population.append(classifier)
                continue
            return match_set

    def generate_prediction_array(self, match_set):
        prediction_array = {}  # prediction array is actually a dictionary...
        fitness_sum_array = {action: 0 for action in action_space}
        for classifier in match_set:
            action = classifier.action
            # todo: is this really necessary?
            if action not in prediction_array:
                prediction_array[action] = classifier.payoff * classifier.fitness
            else:
                prediction_array[action] += classifier.payoff * classifier.fitness
            fitness_sum_array[action] += classifier.fitness
        for action in action_space:
            if fitness_sum_array[action] != 0:
                prediction_array[action] /= fitness_sum_array[action]
        return prediction_array

    def select_action(self, prediction_array):
        if random.random() < exploration_probability:
            return random.choice(list(prediction_array.keys()))
        else:
            action, value = max(prediction_array.items(), key=lambda x: x[1])
            return action

    def generate_action_set(self, match_set, action):
        return {
            classifier
            for classifier in match_set
            if classifier.action == action
        }

    def update_fitness(self, action_set):
        power = 5  # v

        accuracy_vector = []  # k
        for classifier in action_set:
            if classifier.error < error_clip:
                accuracy_vector.append(1)
            else:
                accuracy_vector.append(accuracy_coefficient * ((classifier.error / error_clip) ** power))
        accuracy_sum = sum(accuracy_vector)
        for classifier, accuracy in zip(action_set, accuracy_vector):
            classifier.fitness += learning_rate * (
                    accuracy_sum * classifier.numerosity / accuracy_sum - classifier.fitness
            )

    def do_action_set_subsumption(self, action_set, population):
        subsumer, subsumer_generals = None, None
        for classifier in action_set:
            if classifier.could_subsume():
                generals = classifier.condition.count('#')
                if subsumer is None or generals > subsumer_generals or \
                        (generals == subsumer_generals and random.choice([True, False])):
                    subsumer = classifier
                    subsumer_generals = generals
        if subsumer is not None:
            for classifier in list(action_set):
                if subsumer.is_more_general_than(classifier):
                    subsumer.numerosity += classifier.numerosity
                    action_set.remove(classifier)
                    population.remove(classifier)

    def update_set(self, action_set, payoff, population):
        for c in action_set:
            c.experience += 1
            # update payoff prediction
            if c.experience < 1 / learning_rate:
                c.payoff += (payoff - c.payoff) / c.experience
            else:
                c.payoff += learning_rate * (payoff - c.payoff)
            # update prediction error
            if c.experience < 1 / learning_rate:
                c.error += (abs(payoff - c.payoff) - c.error) / c.experience
            else:
                c.error += learning_rate * abs(payoff - c.payoff)
            # update niche estimate
            if c.experience < 1 / learning_rate:
                c.niche += (sum(i.numerosity for i in action_set) - c.niche) / c.experience
            else:
                c.niche += learning_rate * (sum(i.numerosity for i in action_set) - c.niche)
        self.update_fitness(action_set)
        if do_action_set_subsumption:
            self.do_action_set_subsumption(action_set, population)

    def select_offspring(self, action_set):
        fitness_sum = sum(c.fitness for c in action_set)
        choice_point = random.random() * fitness_sum
        cumulative = 0
        for classifier in action_set:
            cumulative += classifier.fitness
            if cumulative > choice_point:
                return classifier

    def apply_crossover(self, c1, c2):
        x = random.random() * (len(c1.condition) + 1)
        y = random.random() * (len(c1.condition) + 1)
        if x > y:
            x, y = y, x
        i = 0
        c1.condition = [cd2 if x <= i < y else cd1 for cd1, cd2 in zip(c1.condition, c2.condition)]
        c2.condition = [cd1 if x <= i < y else cd2 for cd1, cd2 in zip(c1.condition, c2.condition)]

    def apply_mutation(self, classifier, attributes):
        i = 0
        while True:
            if random.random() < mutation_probability:
                if classifier.condition[i] == '#':
                    classifier.condition[i] = attributes[i]
                else:
                    classifier.condition[i] = '#'
            i += 1
            if i < len(classifier.condition):
                break
        if random.random() < mutation_probability:
            classifier.action = random.choice(list(action_space))

    def insert_in_population(self, classifier, population):
        for c in population:
            if c.condition == classifier.condition and c.action == classifier.action:
                c.numerosity += 1
                return
        population.append(classifier)

    def run_ga(self, action_set, attributes, population):
        sum_last_ga = sum(c.last_in_ga * c.numerosity for c in action_set)
        action_set_numerosity = sum(c.numerosity for c in action_set)
        if self.actual_time - (sum_last_ga / action_set_numerosity) > ga_threshold:
            for classifier in action_set:
                classifier.last_in_ga = self.actual_time
            parent_1 = self.select_offspring(action_set)
            parent_2 = self.select_offspring(action_set)
            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)
            if random.random() < crossover_probability:
                self.apply_crossover(child_1, child_2)
                for child in (child_1, child_2):
                    child.payoff = (parent_1.payoff + parent_2.payoff) / 2
                    child.error = (parent_1.error + parent_2.error) / 2
                    child.fitness = (parent_1.fitness + parent_2.fitness) / 2
            child_1.fitness *= 0.1
            child_2.fitness *= 0.1
            for child in (child_1, child_2):
                self.apply_mutation(child, attributes)
                if do_ga_subsumption:
                    if parent_1.does_subsume(child):
                        parent_1.numerosity += 1
                    elif parent_2.does_subsume(child):
                        parent_2.numerosity += 1
                    else:
                        self.insert_in_population(child, population)
                else:
                    self.insert_in_population(child, population)
                self.delete_from_population(population)

    def run_experiment(self):
        previous_action_set = []  # [A]-1
        previous_reward = 0
        previous_attributes = []

        for attributes, endpoint in zip(train_x, train_y):
            self.actual_time += 1
            match_set = self.generate_match_set(attributes)  # [M]
            prediction_array = self.generate_prediction_array(match_set)  # PA
            action = self.select_action(prediction_array)  # act
            action_set = self.generate_action_set(match_set, action)  # [A]

            # immediate reward
            reward = maximum_reward if action == endpoint else minimum_reward  # ρ
            end_of_problem = True  # this is a

            if len(previous_action_set) > 0:
                payoff = previous_reward * discount_factor * max(prediction_array.values())  # P
                self.update_set(action_set, payoff, self.population)
                self.run_ga(action_set, previous_attributes, self.population)
            if end_of_problem:
                payoff = reward
                self.update_set(action_set, payoff, self.population)
                self.run_ga(action_set, attributes, self.population)
                previous_action_set = []
            else:
                previous_action_set = action_set
                previous_reward = reward
                previous_attributes = attributes

    def fitness(self):
        target = 0
        for attributes, endpoint in zip(train_x, train_y):
            self.actual_time += 1
            match_set = self.generate_match_set(attributes)  # [M]
            prediction_array = self.generate_prediction_array(match_set)  # PA
            action = self.select_action(prediction_array)  # act
            if action == endpoint:
                target += 1
        return target / len(train_x)


def main():
    xcs = XCS()
    for iteration in range(100000):
        xcs.run_experiment()
        print('Iteration:', iteration, 'Fitness:', xcs.fitness(), 'Population size:', len(xcs.population))


if __name__ == '__main__':
    main()
