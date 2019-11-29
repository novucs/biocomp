#include <sstream>
#include <iostream>
#include "Individual.h"
#include "GeneticAlgorithm.h"
#include "Random.h"

Individual::Individual() {}

Individual::Individual(GeneticAlgorithm *ga) : ga(ga) {}

Individual::Individual(GeneticAlgorithm *ga, const std::vector<Rule> &rules) : ga(ga), rules(rules) {}

Individual::Individual(const Individual &individual) : ga(individual.ga), rules(individual.rules) {}

double Individual::generalisation() {
    int generalisation = 0;
    for (Rule &rule : rules) {
        generalisation += rule.generalisation();
    }
    return generalisation == 0 ? 0 : generalisation / (double) rules.size();
}

int Individual::rule_count() {
    return rules.size();
}

Individual Individual::uniform_crossover(Individual &other) {
    std::vector<Rule> new_rules;
    for (int i = 0; i < rules.size(); i++) {
        new_rules.push_back(rules.at(i).uniform_crossover(other.rules.at(i)));
    }
    return Individual(ga, new_rules);
}

Individual Individual::crossover_by_rule(Individual &other) {
    std::vector<Rule> new_rules;
    for (int i = 0; i < rules.size(); i++) {
        Rule rule = rng() < 0.5 ? rules.at(i) : other.rules.at(i);
        new_rules.push_back(rule);
    }
    return Individual(ga, new_rules);
}

Individual Individual::crossover(Individual &other) {
    return rng() < 0.5 ? crossover_by_rule(other) : uniform_crossover(other);
}

Individual Individual::mutate() {
    std::vector<Rule> new_rules;

    // mutate each rule individually
    for (Rule &rule : rules) {
        new_rules.push_back(rule.mutate());
    }

    // randomly swap order of rules
    for (int i = 0; i < (rules.size() - 1); i++) {
        if (ga->should_mutate()) {
            std::iter_swap(new_rules.begin() + i, new_rules.begin() + i + 1);
        }
    }

    return Individual(ga, new_rules);
}

int Individual::evaluate(std::vector<double> &features) {
    for (Rule &rule : rules) {
        if (rule.matches(features)) {
            return rule.get_action();
        }
    }
    return -1; // force classifier to learn comprehensive rule set
}

int Individual::correct_count(Dataset &dataset) {
    int correct_count = 0;
    for (int i = 0; i < dataset.features.size(); i++) {
        if (evaluate(dataset.features.at(i)) == dataset.labels.at(i)) {
            correct_count += 1;
        }
    }
    return correct_count;
}

double Individual::fitness(Dataset &dataset) {
    double correctness_factor = correct_count(dataset) / (double) dataset.features.size();
    double generalisation_factor = generalisation() / (double) dataset.features.size();
    return correctness_factor + generalisation_factor;
}

std::string Individual::dump() {
    std::string dump;
    for (Rule &rule : rules) {
        dump += rule.dump();
        dump += "|";
    }
    return dump.substr(0, dump.size() - 1);
}

Individual Individual::remove_rule() {
    std::vector<Rule> new_rules;
    int index = ((double) rules.size()) * rng();
    for (int i = 0; i < rules.size(); i++) {
        if (i == index) {
            continue;
        }
        new_rules.push_back(rules.at(i));
    }
    return Individual(ga, new_rules);
}

bool Individual::is_subsumed(int rule_index) {
    for (int i = 0; i < rule_index; i++) {
        if (rules.at(i).subsumes(rules.at(rule_index))) {
            return true;
        }
    }
    return false;
}

Individual Individual::compress() {
    std::vector<Rule> new_rules;
    for (int i = 0; i < rules.size(); i++) {
        Rule &rule = rules.at(i);
        if (is_subsumed(i)) {
            continue;
        }
        new_rules.push_back(rule);
    }
    return Individual(ga, new_rules);
}

Individual Individual::cover(Dataset &dataset, std::vector<std::vector<int>> &wrong_classifications) {
    if (wrong_classifications.empty()) {
        return Individual(*this);
    }

    std::vector<int> indices = wrong_classifications.at(rng() * (double) wrong_classifications.size());
    int sample_id = indices.at(0);
    int rule_id = indices.at(1);

    std::vector<Rule> new_rules(this->rules);
    new_rules.erase(new_rules.begin() + rule_id);

    Individual individual(ga, new_rules);
    individual = individual.compress();

    // cover missing rules using dataset samples this individual classifies wrongly
    Rule rule = rule_from_sample(ga, dataset.features.at(sample_id), dataset.labels.at(sample_id));
    individual.rules.insert(individual.rules.begin(), rule);

    // cover remaining missing rules using random dataset samples
    int missing_rules = ga->get_rule_count() - individual.rule_count();
    for (int i = 0; i < missing_rules; i++) {
        int index = wrong_classifications.at(rng() * (double) wrong_classifications.size()).at(0);
        rule = rule_from_sample(ga, dataset.features.at(index), dataset.labels.at(index));
        individual.rules.insert(individual.rules.begin(), rule);
    }

    return individual;
}

std::vector<std::vector<int>> Individual::wrong_classifications(Dataset &dataset) {
    std::vector<std::vector<int>> classifications;
    for (int sample_id = 0; sample_id < dataset.features.size(); sample_id++) {
        for (int rule_id = 0; rule_id < rules.size(); rule_id++) {
            Rule &rule = rules.at(rule_id);
            if (rule.matches(dataset.features.at(sample_id))) {
                if (rule.get_action() != dataset.labels.at(sample_id)) {
                    classifications.push_back({sample_id, rule_id});
                }
                break;
            }
        }
    }
    return classifications;
}

Individual dummy_individual() {
    return Individual();
}

Individual generate_individual(GeneticAlgorithm *ga) {
    std::vector<Rule> new_rules;
    for (int i = 0; i < ga->get_rule_count(); i++) {
        new_rules.push_back(generate_rule(ga));
    }
    return Individual(ga, new_rules);
}

Individual load_individual(GeneticAlgorithm *ga, std::string dump) {
    std::vector<Rule> new_rules;
    std::stringstream ss(dump);
    for (int i = 0; i < ga->get_rule_count(); i++) {
        std::string substr;
        std::getline(ss, substr, '|');
        new_rules.push_back(load_rule(ga, substr));
    }
    return Individual(ga, new_rules);
}

Individual
individual_from_samples(GeneticAlgorithm *ga, Dataset &dataset) {
    std::vector<Rule> new_rules;
    if (ga->get_rule_count() == dataset.features.size()) {
        // load 1 to 1 mapping of rules to dataset features for instant 100% fitness at this rule count
        for (int i = 0; i < dataset.features.size(); i++) {
            Rule rule = rule_from_sample(ga, dataset.features.at(i), dataset.labels.at(i));
            new_rules.push_back(rule);
        }
    } else {
        for (int i = 0; i < ga->get_rule_count(); i++) {
            int index = rng() * dataset.features.size();
            Rule rule = rule_from_sample(ga, dataset.features.at(index), dataset.labels.at(index));
            new_rules.push_back(rule);
        }
    }
    return Individual(ga, new_rules);
}
