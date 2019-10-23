import random

from biocomp import datasets

rule_base_size = 10  # N
learning_rate = 0.01  # β
genetic_algorithm_run_chance = 0.1  # g
mutation_chance = 0.0125  # μ
mutation_wildcard_change_chance = 0.0125  # p#
crossover_chance = 0.75  # χ

# environment
train_x, train_y, *_ = datasets.split(datasets.load_dataset_1())


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

    def fitness(self):
        return 1 / (self.error + 1)

    def matches(self, attributes):
        return all(c == a or c == '#' for c, a in zip(self.condition, attributes))


class YCS:  # Y Learning Classifier System
    def __init__(self):
        self.rulebase = [
            Rule(('#' if random.random() < mutation_chance else a for a in x), y)
            for _, x, y in zip(range(rule_base_size), train_x, train_y)
        ]

    def roulette_wheel_selection(self, population):
        total = sum(rule.fitness() for rule in population)
        point = random.uniform(0, total)
        spin = 0
        for rule in population:
            spin += rule.fitness()
            if spin >= point:
                return rule

    def single_point_crossover(self, mum, dad):
        if random.random() > crossover_chance:
            return mum.copy()
        point = random.randrange(len(dad))
        return mum[:point] + dad[point:]

    def train(self):
        # indicates whether to explore or exploit, alternates during training
        # todo: might be kinda cool to see if curiosity can be applied here?
        #  would only make sense for complex sensorimotor streams such as in an
        #  RL setting though...
        explore = False

        for attributes, endpoint in zip(train_x, train_y):
            match_set = {rule for rule in self.rulebase if rule.matches(attributes)}

            # select action
            explore = not explore  # alternate between exploring and exploiting
            if explore:
                # explore: take random action
                action = random.choice([0, 1])
            else:
                # exploit: take action with maximum average payoff
                payoffs = {}
                for rule in match_set:
                    payoffs[rule.action] = payoffs.get(rule.action) + rule.payoff
                action = max(payoffs.items(), key=lambda p: p[1])[0]

            action_set = {rule for rule in match_set if rule.action == action}

            # reinforcement using Widrow-Hoff delta rule
            # calculate immediate reward
            # todo: determine immediate reward payoff implementation
            reward = 1000 if endpoint == action else 0  # P
            for rule in action_set:
                rule.error += learning_rate * (abs(reward - rule.payoff) - rule.error)
                rule.niche_factor += learning_rate * (len(action_set) - rule.niche_factor)
                rule.payoff += learning_rate * (reward - rule.payoff)

            # genetic algorithm
            if explore and random.random() < genetic_algorithm_run_chance:
                # selection
                # todo: confirm which rule set the parents are selected from
                mum = self.roulette_wheel_selection(action_set)
                dad = self.roulette_wheel_selection(action_set)

                # crossover
                offspring_rule_1 = self.single_point_crossover(mum.rule, dad.rule)
                offspring_rule_2 = self.single_point_crossover(dad.rule, mum.rule)

                # mutation
