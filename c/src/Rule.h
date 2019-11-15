#ifndef C_RULE_H
#define C_RULE_H

#include <vector>
#include <string>
#include "Bounds.h"

class GeneticAlgorithm;

class Rule {
private:
    GeneticAlgorithm *ga;
    std::vector<Bounds> *condition;
    int action;
public:

    Rule(GeneticAlgorithm *ga, std::vector<Bounds> *condition, int action);

    ~Rule();

    int get_action();

    double generalisation();

    Rule *uniform_crossover(Rule *other);

    Rule *mutate();

    Rule *copy();

    bool matches(std::vector<double> *features);

    std::string dump();

    bool subsumes(Rule other);
};

Rule *generate_rule(GeneticAlgorithm *ga);

Rule *load_rule(GeneticAlgorithm *ga, std::string dump);

Rule *rule_from_sample(GeneticAlgorithm *ga, std::vector<double> *features, int label);

#endif //C_RULE_H
