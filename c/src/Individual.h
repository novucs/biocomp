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
    Individual();

    Individual(GeneticAlgorithm *ga);

    Individual(GeneticAlgorithm *ga, const std::vector<Rule> &rules);

    Individual(const Individual &individual);

    double generalisation();

    int rule_count();

    Individual crossover(Individual &other);

    Individual mutate();

    int evaluate(std::vector<double> &features);

    int correct_count(Dataset &dataset);

    double fitness(Dataset &dataset);

    std::string dump();

    Individual remove_rule();

    Individual remove_rules(int count);

    bool is_subsumed(int rule_index);

    Individual compress();

    Individual cover(Dataset &dataset, std::vector<std::vector<int>> &wrong_classifications);

    std::vector<std::vector<int>> wrong_classifications(Dataset &dataset);
};

Individual dummy_individual();

Individual generate_individual(GeneticAlgorithm *ga, int rule_count);

Individual load_individual(GeneticAlgorithm *ga, std::string dump, int rule_count);

Individual individual_from_samples(GeneticAlgorithm *ga, Dataset &dataset, int rule_count);

#endif //C_INDIVIDUAL_H
