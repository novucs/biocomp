#ifndef C_INDIVIDUAL_H
#define C_INDIVIDUAL_H

#include "Rule.h"

class GeneticAlgorithm;

class Individual {
private:
    GeneticAlgorithm *ga;
    std::vector<Rule> rules = std::vector<Rule>();
public:
    Individual(GeneticAlgorithm *ga);

    Individual(GeneticAlgorithm *ga, const std::vector<Rule> &rules);

    double generalisation();

    int rule_count();

    Individual uniform_crossover(Individual &other);

    Individual crossover_by_rule(Individual &other);

    Individual crossover(Individual &other);

    Individual mutate();

    int evaluate(std::vector<double> &features);

    int correct_count(std::vector<std::vector<double>> &features, std::vector<int> &labels);

    double fitness(std::vector<std::vector<double>> &features, std::vector<int> &labels);

    std::string dump();

    Individual remove_rule();

    bool is_subsumed(int rule_index);

    Individual compress();

    Individual cover(std::vector<std::vector<double>> &features, std::vector<int> &labels);
};

Individual generate_individual(GeneticAlgorithm *ga);

Individual load_individual(GeneticAlgorithm *ga, std::string dump);

Individual
individual_from_samples(GeneticAlgorithm *ga, std::vector<std::vector<double>> &features, std::vector<int> &labels);

#endif //C_INDIVIDUAL_H
