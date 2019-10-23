import random

from biocomp import datasets

rule_base_size = 10  # N
learning_rate = 0.01  # β
# todo: determine true immediate reward payoff implementation
immediate_reward_payoff = 1  # P
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

    @staticmethod
    def from_uniform_crossover(mother, father):
        # coin flip between swapping specification and generalisation
        condition = [
            random.choice([m, f]) if m == '#' or f == '#' else m
            for m, f in zip(mother.condition, father.condition)
        ]
        action = mother.action
        return Rule(condition, action)

    @staticmethod
    def mutate(rule, training_instance):
        # coin flip between a wildcard or whats in the training instance
        # if the mutation check passes, otherwise retain the original gene
        condition = [
            random.choice(['#', a]) if random.random() < mutation_wildcard_change_chance else c
            for c, a in zip(rule.condition, training_instance.attributes)
        ]
        action = rule.action
        return Rule(condition, action)


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
            for rule in action_set:
                rule.error += learning_rate * (
                            abs(immediate_reward_payoff - rule.payoff) - rule.error)
                rule.niche_factor += learning_rate * (len(action_set) - rule.niche_factor)
                rule.payoff += learning_rate * (immediate_reward_payoff - rule.payoff)

            # genetic algorithm
            if explore and random.random() < genetic_algorithm_run_chance:
                # selection
                mother = self.roulette_wheel_selection(action_set)
                father = self.roulette_wheel_selection(action_set)

                # crossover

