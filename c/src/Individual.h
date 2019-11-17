#ifndef C_INDIVIDUAL_H
#define C_INDIVIDUAL_H

#include "Rule.h"
#include "Dataset.h"

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

    int correct_count(Dataset &dataset);

    double fitness(Dataset &dataset);

    std::string dump();

    Individual remove_rule();

    bool is_subsumed(int rule_index);

    Individual compress();

    Individual cover(Dataset &dataset);
};

Individual generate_individual(GeneticAlgorithm *ga);

Individual load_individual(GeneticAlgorithm *ga, std::string dump);

Individual
individual_from_samples(GeneticAlgorithm *ga, Dataset &dataset);

#endif //C_INDIVIDUAL_H
