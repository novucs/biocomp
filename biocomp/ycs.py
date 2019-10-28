import itertools
import random

from biocomp import datasets

rule_base_size = 60  # N
learning_rate = 0.2  # β
ga_chance = 0.25  # g
mutation_chance = 0.0125  # μ
generalisation_rate = 0.0125  # p#
crossover_chance = 0.75  # χ
# todo: understand why payoffs are different for each case in larry's
#  implementation, when it could just be something like:
#  payoff = 1000 if correct else 0
rewards = {  # P
    # key:
    #  (input, action): payoff
    (1, 1): 1000,
    (1, 0): 800,
    (0, 1): 1000,
    (0, 0): 3000,
    # todo: input may never be '#' in our example, is this here for partially
    #  observable environments, or something else entirely?
    ('#', 0): 1900,
    ('#', 1): 1000,
}

# environment
train_x, train_y, *_ = datasets.split(datasets.load_dataset_1())


def roulette_wheel_selection(population, key=lambda i: i.fitness):
    total = sum(key(individual) for individual in population)
    point = random.uniform(0, total)
    spin = 0
    for index, individual in enumerate(population):
        spin += key(individual)
        if spin >= point:
            return individual, index


def single_point_crossover(mum, dad):
    if random.random() > crossover_chance:
        return mum.copy()
    point = random.randrange(len(dad))
    return mum[:point] + dad[point:]


def mutate(condition):
    return [
        '#' if random.random() < generalisation_rate else c
        for c in condition
    ]


class Rule:
    def __init__(self, condition, action):
        # the condition that matches a training instances attributes
        self.condition = random.choices([0, 1, '#'], k=len(train_x[0]))

        # the predicted classification for training instances that match this rule
        self.action = random.choice([0, 1])

        # predicted payoff (ρ)
        self.payoff = 10.0

        # the error (ε) in the the rule's predicted payoff
        self.error = 10.0

        # average size of niches (action sets) in which the rule participates (σ)
        self.niche_factor = 10.0

    @property
    def fitness(self):
        return 1 / (self.error + 1)

    def matches(self, attributes):
        return all(c == a or c == '#' for c, a in zip(self.condition, attributes))


def greedy_act(match_set):
    votes = {}
    for rule in match_set:
        votes[rule.action] = votes.get(rule.action, 0) + rule.payoff
    action = max(votes.items(), key=lambda p: p[1])[0]
    return action


class YCS:  # Y Learning Classifier System
    def __init__(self):
        self.rulebase = [
            Rule(mutate(x), y)
            for _, x, y in zip(range(rule_base_size), train_x, train_y)
        ]

    def replace_into_population(self, individual):
        old, index = roulette_wheel_selection(
            self.rulebase, key=lambda i: 1 / i.niche_factor)
        self.rulebase[index] = individual

    def accuracy(self):
        correct = 0
        for attributes, endpoint in zip(train_x, train_y):
            match_set = {rule for rule in self.rulebase if rule.matches(attributes)}
            if len(match_set) == 0:
                correct += random.randint(0, 1)
                continue
            action = greedy_act(match_set)
            if action == endpoint:
                correct += 1
        return correct / len(train_x)

    def train(self):
        # indicates whether to explore or exploit, alternates during training
        # todo: might be kinda cool to see if curiosity can be applied here?
        #  would only make sense for complex sensorimotor streams such as in an
        #  RL setting though...
        explore = False

        for attributes, endpoint in zip(train_x, train_y):
            match_set = {rule for rule in self.rulebase if rule.matches(attributes)}

            if len(match_set) == 0:
                # covering
                condition = mutate(attributes)
                # todo: why did larry suggest using a random action instead of
                #  the endpoint?
                # action = endpoint
                action = random.choice((0, 1))
                individual = Rule(condition, action)
                self.replace_into_population(individual)
                continue

            # select action
            explore = not explore  # alternate between exploring and exploiting
            if explore:
                # explore: take random action
                action = random.choice([0, 1])
            else:
                # exploit: take action with maximum average payoff
                action = greedy_act(match_set)

            action_set = {rule for rule in match_set if rule.action == action}

            # reinforcement using Widrow-Hoff delta rule
            # calculate immediate reward
            # reward = rewards[(endpoint, action)]  # P
            reward = 1000 if endpoint == action else 0  # P
            for rule in action_set:
                # todo: the parameters here do not seem very well normalised so
                #  updates are probably going to be quite unstable
                error = learning_rate * (abs(reward - rule.payoff) - rule.error)
                niche = learning_rate * (len(action_set) - rule.niche_factor)
                payoff = learning_rate * (reward - rule.payoff)
                # print('Error:', error, '\tNiche:', niche, '\tPayoff:', payoff)
                # print('Error:', rule.error, '\tNiche', rule.niche_factor,
                #       '\tPayoff:', rule.payoff)
                rule.error += error
                rule.niche_factor += niche
                rule.payoff += payoff

            # genetic algorithm
            # todo: should this genetic algorithm be ran under a different
            #  context? i.e. outside the environment sampling loop
            if explore and len(action_set) > 0 and random.random() < ga_chance:
                # selection
                # todo: confirm which rule set the parents are selected from
                mum, _ = roulette_wheel_selection(action_set)
                dad, _ = roulette_wheel_selection(action_set)

                for mum, dad in itertools.permutations((mum, dad), 2):
                    condition = single_point_crossover(mum.condition, dad.condition)
                    # todo: check whether mutation chance should be applied per attribute
                    if random.random() < mutation_chance:
                        condition = mutate(condition)
                    offspring = Rule(condition, action)
                    self.replace_into_population(offspring)


def main():
    ycs = YCS()
    for iteration in range(1000):
        ycs.train()
        print('Iteration:', iteration, 'Accuracy:', ycs.accuracy())


if __name__ == '__main__':
    main()
