#include <sstream>
#include "Individual.h"
#include "GeneticAlgorithm.h"
#include "Random.h"

Individual::Individual(GeneticAlgorithm *ga) {
    this->ga = ga;
    this->rules = new std::vector<Rule *>();
}

Individual::Individual(GeneticAlgorithm *ga, std::vector<Rule *> *rules) {
    this->ga = ga;
    this->rules = rules;
}

Individual::~Individual() {
    for (Rule *rule : *rules) {
        delete rule;
    }
    delete rules;
}

double Individual::generalisation() {
    int generalisation = 0;
    for (Rule *rule : *rules) {
        generalisation += rule->generalisation();
    }
    return generalisation == 0 ? 0 : generalisation / (double) rules->size();
}

int Individual::rule_count() {
    return rules->size();
}

Individual *Individual::uniform_crossover(Individual *other) {
    auto *new_rules = new std::vector<Rule *>();
    for (int i = 0; i < rules->size(); i++) {
        new_rules->push_back(rules->at(i)->uniform_crossover(other->rules->at(i)));
    }
    return new Individual(ga, new_rules);
}

Individual *Individual::crossover_by_rule(Individual *other) {
    auto *new_rules = new std::vector<Rule *>();
    for (int i = 0; i < rules->size(); i++) {
        new_rules->push_back(rng() < 0.5 ? rules->at(i)->copy() : other->rules->at(i)->copy());
    }
    return new Individual(ga, new_rules);
}

Individual *Individual::crossover(Individual *other) {
    return rng() < 0.5 ? crossover_by_rule(other) : uniform_crossover(other);
}

Individual *Individual::mutate() {
    auto *new_rules = new std::vector<Rule *>();

    // mutate each rule individually
    for (Rule *rule : *rules) {
        new_rules->push_back(rule->mutate());
    }

    // randomly swap order of rules
    for (int i = 0; i < (rules->size() - 1); i++) {
        if (rng() < ga->mutation_chance()) {
            std::iter_swap(new_rules->begin() + i, new_rules->begin() + i + 1);
        }
    }

    return new Individual(ga, new_rules);
}

int Individual::evaluate(std::vector<double> *features) {
    for (Rule *rule : *rules) {
        if (rule->matches(features)) {
            return rule->get_action();
        }
    }
    return -1; // force classifier to learn comprehensive rule set
}

int Individual::correct_count(std::vector<std::vector<double>> *features, std::vector<int> *labels) {
    int correct_count = 0;
    for (int i = 0; i < features->size(); i++) {
        if (evaluate(&features->at(i)) == labels->at(i)) {
            correct_count += 1;
        }
    }
    return correct_count;
}

double Individual::fitness(std::vector<std::vector<double>> *features, std::vector<int> *labels) {
    double correctness_factor = correct_count(features, labels) / (double) features->size();
    double generalisation_factor = generalisation() / (double) features->size();
    return correctness_factor + generalisation_factor;
}

Individual *Individual::copy() {
    auto *new_rules = new std::vector<Rule *>();
    for (Rule *rule : *rules) {
        new_rules->push_back(rule->copy());
    }
    return new Individual(ga, new_rules);
}

std::string Individual::dump() {
    std::string dump;
    for (Rule *rule : *rules) {
        dump += rule->dump();
        dump += "|";
    }
    return dump.substr(0, dump.size() - 1);
}

Individual *Individual::remove_rule() {
    auto *individual = copy();
    int index = individual->rules->size() * rng();
    individual->rules->erase(individual->rules->begin() + index);
    return individual;
}


bool Individual::is_subsumed(int rule_index) {
    for (int i = 0; i < rule_index; i++) {
        if (rules->at(i)->subsumes(rules->at(rule_index))) {
            return true;
        }
    }
    return false;
}


Individual *Individual::compress() {
    auto *new_rules = new std::vector<Rule *>();
    for (int i = 0; i < rules->size(); i++) {
        Rule *rule = rules->at(i);
        if (is_subsumed(i)) {
            continue;
        }
        new_rules->push_back(rule->copy());
    }
    return new Individual(ga, new_rules);
}

Individual *Individual::cover(std::vector<std::vector<double>> *features, std::vector<int> *labels) {
    std::vector<int> wrong_classifications;
    for (int i = 0; i < features->size(); i++) {
        if (evaluate(&features->at(i)) != labels->at(i)) {
            wrong_classifications.push_back(i);
        }
    }

    auto *individual = compress();

    // cover missing rules using dataset samples this individual classifies wrongly
    int missing_rules = ga->get_rule_count() - individual->rule_count();
    int wrong_covering_count = std::min((int) wrong_classifications.size(), missing_rules);
    for (int i = 0; i < wrong_covering_count; i++) {
        int index = wrong_classifications.at(i);
        Rule *rule = rule_from_sample(ga, &features->at(index), labels->at(index));
        individual->rules->insert(individual->rules->begin(), rule);
    }

    // cover remaining missing rules using random dataset samples
    missing_rules = ga->get_rule_count() - individual->rule_count();
    for (int i = 0; i < missing_rules; i++) {
        int index = rng() * features->size();
        Rule *rule = rule_from_sample(ga, &features->at(index), labels->at(index));
        individual->rules->insert(individual->rules->begin(), rule);
    }

    return individual;
}

Individual *generate_individual(GeneticAlgorithm *ga) {
    auto *rules = new std::vector<Rule *>();
    for (int i = 0; i < ga->get_rule_count(); i++) {
        rules->push_back(generate_rule(ga));
    }
    return new Individual(ga, rules);
}

Individual *load_individual(GeneticAlgorithm *ga, std::string dump) {
    auto *rules = new std::vector<Rule *>();
    std::stringstream ss(dump);
    for (int i = 0; i < ga->get_rule_count(); i++) {
        std::string substr;
        std::getline(ss, substr, '|');
        rules->push_back(load_rule(ga, substr));
    }
    return new Individual(ga, rules);
}

Individual *
individual_from_samples(GeneticAlgorithm *ga, std::vector<std::vector<double>> *features, std::vector<int> *labels) {
    auto *rules = new std::vector<Rule *>();
    if (ga->get_rule_count() == features->size()) {
        // load 1 to 1 mapping of rules to dataset features for instant 100% fitness at this rule count
        for (int i = 0; i < features->size(); i++) {
            Rule *rule = rule_from_sample(ga, &features->at(i), labels->at(i));
            rules->push_back(rule);
        }
    } else {
        for (int i = 0; i < ga->get_rule_count(); i++) {
            int index = rng() * features->size();
            Rule *rule = rule_from_sample(ga, &features->at(index), labels->at(index));
            rules->push_back(rule);
        }
    }
    return new Individual(ga, rules);
}
