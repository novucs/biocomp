#include "Rule.h"

#include <cmath>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include "GeneticAlgorithm.h"
#include "Random.h"

Rule::Rule(GeneticAlgorithm *ga, const std::vector<Bounds> &condition, int action) : ga(ga), condition(condition),
                                                                                     action(action) {}

Rule::Rule(const Rule &rule) : ga(rule.ga), condition(rule.condition), action(rule.action) {}

double Rule::generalisation() {
    int hash_count = 0;
    for (Bounds &bounds : condition) {
        if (bounds.is_wildcard()) {
            hash_count += 1;
        }
    }
    return hash_count / (double) condition.size();
}

Rule Rule::mutate() {
    std::vector<Bounds> new_condition;
    for (Bounds &bounds : condition) {
        new_condition.push_back(bounds.mutate());
    }
    int new_action = ga->should_mutate() ? (int) std::round(rng()) : action;
    return Rule(ga, new_condition, new_action);
}

bool Rule::matches(std::vector<double> &features) {
    for (int i = 0; i < features.size(); i++) {
        if (!condition.at(i).contains(features.at(i))) {
            return false;
        }
    }
    return true;
}

std::string Rule::dump() {
    std::string dump;
    for (Bounds &bounds : condition) {
        dump += bounds.dump();
        dump += ",";
    }
    dump += std::to_string(action);
    return dump;
}

bool Rule::subsumes(Rule &other) {
    for (int i = 0; i < condition.size(); i++) {
        if (!condition.at(i).subsumes(other.condition.at(i))) {
            return false;
        }
    }
    return true;
}

int Rule::get_action() {
    return action;
}

Rule generate_rule(GeneticAlgorithm *ga) {
    std::vector<Bounds> new_condition;
    for (int i = 0; i < ga->get_condition_size(); i++) {
        new_condition.push_back(random_bounds(ga));
    }
    int action = std::round(rng());
    return Rule(ga, new_condition, action);
}

Rule load_rule(GeneticAlgorithm *ga, std::string dump) {
    std::vector<Bounds> new_condition;
    std::stringstream ss(dump);

    for (int i = 0; i < ga->get_condition_size(); i++) {
        std::string substr;
        std::getline(ss, substr, ',');
        new_condition.push_back(load_bounds(ga, substr));
    }

    int action;
    ss >> action;

    return Rule(ga, new_condition, action);
}

Rule rule_from_sample(GeneticAlgorithm *ga, std::vector<double> &features, int label) {
    std::vector<Bounds> new_condition;
    for (double feature : features) {
        new_condition.push_back(random_bounds(ga, feature));
    }
    return Rule(ga, new_condition, label);
}
