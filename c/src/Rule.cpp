#include "Rule.h"

#include <utility>
#include <cmath>
#include <sstream>
#include "GeneticAlgorithm.h"


Rule::Rule(GeneticAlgorithm *ga, std::vector<Bounds> *condition, int action) {
    this->ga = ga;
    this->condition = condition;
    this->action = action;
}

Rule::~Rule() {
    delete condition;
}

double Rule::generalisation() {
    int hash_count = 0;
    for (Bounds bounds : *condition) {
        if (bounds.is_generalisable()) {
            hash_count += 1;
        }
    }
    return hash_count / condition->size();
}

Rule *Rule::uniform_crossover(Rule *other) {
    auto *new_condition = new std::vector<Bounds>();
    for (std::vector<Bounds>::size_type i = 0; i < condition->size(); i++) {
        new_condition->push_back(std::rand() < 0.5 ? condition->at(i) : other->condition->at(i));
    }
    int new_action = std::rand() < 0.5 ? action : other->action;
    return new Rule(ga, new_condition, new_action);
}

Rule *Rule::mutate() {
    auto *new_condition = new std::vector<Bounds>();
    for (Bounds bounds : *condition) {
        new_condition->push_back(*bounds.mutate());
    }
    int new_action = std::rand() < ga->mutation_chance() ? (int) std::round(std::rand()) : action;
    return new Rule(ga, new_condition, new_action);
}

Rule *Rule::copy() {
    auto *new_condition = new std::vector<Bounds>(*condition);
    return new Rule(ga, new_condition, action);
}

bool Rule::matches(std::vector<double> *features) {
    for (int i = 0; i < features->size(); i++) {
        if (!condition->at(i).contains(features->at(i))) {
            return false;
        }
    }
    return true;
}

std::string Rule::dump() {
    std::string dump;
    for (Bounds bounds : *condition) {
        dump += bounds.dump();
        dump += ",";
    };
    return dump.substr(0, dump.size() - 1);
}

bool Rule::subsumes(Rule other) {
    for (int i = 0; i < condition->size(); i++) {
        if (!condition->at(i).subsumes(other.condition->at(i))) {
            return false;
        }
    }
    return true;
}

int Rule::get_action() {
    return action;
}

Rule *generate_rule(GeneticAlgorithm *ga) {
    auto *condition = new std::vector<Bounds>();
    for (int i = 0; i < ga->get_condition_size(); i++) {
        condition->push_back(*random_bounds(ga));
    }
    int action = std::round(std::rand());
    return new Rule(ga, condition, action);
}

Rule *load_rule(GeneticAlgorithm *ga, std::string dump) {
    auto *condition = new std::vector<Bounds>();
    std::stringstream ss(dump);

    for (int i = 0; i < ga->get_condition_size(); i++) {
        std::string substr;
        std::getline(ss, substr, ',');
        condition->push_back(*load_bounds(ga, substr));
    }

    int action;
    ss >> action;

    return new Rule(ga, condition, action);
}

Rule *rule_from_sample(GeneticAlgorithm *ga, std::vector<double> *features, int label) {
    auto *condition = new std::vector<Bounds>();
    for (double feature : *features) {
        condition->push_back(*random_bounds(ga, feature));
    }
    return new Rule(ga, condition, label);
}
