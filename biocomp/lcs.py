# I'm doing my research now...
# genetic algorithms
# learning classifier systems
# dataset: attributes -> endpoint
# michigan-style rules: condition -> action
# todo: look into hybrid lcs and pittsburgh-style lcs
import itertools
import random
from collections import defaultdict

from biocomp import datasets


class TrainingInstance:
    def __init__(self, attributes, endpoint):
        self.attributes = attributes
        self.endpoint = endpoint


class Rule:
    def __init__(self, condition, action):
        self.condition = condition
        self.action = action

    @staticmethod
    def from_training_instance(training_instance):
        condition = [
            random.choice([attribute, '#'])
            for attribute in training_instance.attributes
        ]
        action = training_instance.endpoint
        return Rule(condition, action)

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
    def mutate(rule, training_instance, rate=0.0125):
        # coin flip between a wildcard or whats in the training instance
        # if the mutation check passes, otherwise retain the original gene
        condition = [
            random.choice(['#', a]) if random.random() < rate else c
            for c, a in zip(rule.condition, training_instance.attributes)
        ]
        action = rule.action
        return Rule(condition, action)

    def matches(self, attributes):
        return all(c == a or c == '#' for c, a in zip(self.condition, attributes))


class Classifier:
    def __init__(self, rule, birth_iteration):
        self.rule = rule

        # number of copies of this classifier that exist in the population
        self.numerosity = 1

        # the amount of times this classifer has been in a match set
        self.match_count = 1

        # the amount of times this classifer has been in a correct set
        self.correct_count = 1

        # when this classifier was first made
        self.birth_iteration = birth_iteration

        # not sure yet
        self.average_match_set_size = 0

    def accuracy(self):
        return self.correct_count / self.match_count

    def fitness(self):
        return self.accuracy() ** 5

    def subsumes(self, other):
        if self.accuracy() < other.accuracy():
            return False
        return self.rule.matches(other.rule.condition)


class LearningClassifierSystem:
    """ Implementation of a Michigan-style LCS. """

    def __init__(self, environment):
        self.environment = environment
        self.population = []
        self.iterations = 1000
        self.tournament_size = 5

    def tournament_selection(self, correct):
        tournament = random.choices(correct, k=self.tournament_size)
        winner = max(tournament, key=lambda classifier: classifier.fitness())
        return winner

    def learn(self):
        for iteration in range(self.iterations):
            for training_instance in self.environment:
                self.learn_from_training_instance(iteration, training_instance)
            print('Iteration:', iteration, 'Fitness:', self.fitness(),
                  'Population Size:', len(self.population))

    def predict(self, attributes):
        matches = [
            classifier
            for classifier in self.population
            if classifier.rule.matches(attributes)
        ]

        votes = defaultdict(int)
        for classifier in matches:
            votes[classifier.rule.action] += classifier.fitness() * classifier.numerosity

        prediction = max(votes.items(), key=lambda i: i[1])[0]
        return prediction

    def fitness(self):
        return sum(
            1
            for training_instance in self.environment
            if self.predict(training_instance.attributes) == training_instance.endpoint
        ) / len(self.environment)

    def roulette_wheel_deletion(self):
        def probability(classifier):
            # keeps population from being overrun by few rules with large numerosities
            return classifier.numerosity / classifier.fitness()
            # return 1 / classifier.fitness()

        total = sum(probability(classifier) for classifier in self.population)
        index = random.uniform(0, total)
        upto = 0
        for classifier in self.population:
            upto += probability(classifier)
            if upto >= index:
                self.population.remove(classifier)
                return

    def learn_from_training_instance(self, iteration, training_instance):
        # matching
        matches = [
            classifier
            for classifier in self.population
            if classifier.rule.matches(training_instance.attributes)
        ]

        correct = [
            classifier
            for classifier in matches
            if classifier.rule.action == training_instance.endpoint
        ]

        # covering
        if len(correct) == 0:
            rule = Rule.from_training_instance(training_instance)
            classifier = Classifier(rule, iteration)
            self.population.append(classifier)

        # parameter updates
        for classifier in matches:
            classifier.match_count += 1

        for classifier in correct:
            classifier.correct_count += 1

        # subsumption
        self.subsumption(correct)

        if len(correct) > 0:
            # tournament selection
            mother = self.tournament_selection(correct).rule
            father = self.tournament_selection(correct).rule

            # uniform crossover
            offspring_a = Rule.from_uniform_crossover(mother, father)
            offspring_b = Rule.from_uniform_crossover(father, mother)

            # mutation
            offspring_a = Rule.mutate(offspring_a, training_instance)
            offspring_b = Rule.mutate(offspring_b, training_instance)

            # subsumption
            # studies suggest only using the GA subsumption is more conservative
            # with less risk of degrading learning performance
            classifier_a = Classifier(offspring_a, iteration)
            classifier_b = Classifier(offspring_b, iteration)
            self.population.extend((classifier_a, classifier_b))
            self.subsumption((classifier_a, classifier_b))

        # deletion
        max_population_size = sum(classifier.numerosity for classifier in self.population)
        number_of_deletions = max(0, len(self.population) - max_population_size)
        for deletion in range(number_of_deletions):
            self.roulette_wheel_deletion()

    def subsumption(self, correct):
        for classifier_a, classifier_b in itertools.permutations(correct, 2):
            if classifier_b in self.population and classifier_a.subsumes(classifier_b):
                self.population.remove(classifier_b)
                self.population.append(classifier_a)
                classifier_a.numerosity += 1


def main():
    environment = [TrainingInstance(attributes, endpoint)
                   for attributes, endpoint in datasets.load_dataset_1()]
    lcs = LearningClassifierSystem(environment)
    lcs.learn()


if __name__ == '__main__':
    main()
