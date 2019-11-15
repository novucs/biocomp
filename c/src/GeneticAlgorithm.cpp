#include "GeneticAlgorithm.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <gmpxx.h>
#include <sstream>
#include <bits/unordered_map.h>
#include "Individual.h"

GeneticAlgorithm::GeneticAlgorithm() {
    Dataset glob("../../datasets/2019/data3.txt");
    datasets = glob.split({0.333, 0.333, 0.333});
    this->train = &datasets->at(0);
    this->cross_validation = &datasets->at(1);
    this->test = &datasets->at(2);
}

GeneticAlgorithm::~GeneticAlgorithm() {
    delete datasets;
}

double GeneticAlgorithm::mutation_chance() {
    return 0.00125;
}

int GeneticAlgorithm::get_condition_size() {
    return train->features->at(0).size();
}

int GeneticAlgorithm::get_rule_count() {
    if (rule_count == 0) {
        rule_count = std::min((int) train->features->size(), 30);
    }
    return rule_count;
}

void GeneticAlgorithm::load_population(std::string filename) {
    std::ifstream datafile(filename);

    Individual *best;
    int best_rule_count = std::numeric_limits<int>::max();
    double best_fitness = 0;

    for (std::string line; getline(datafile, line);) {
        std::stringstream ss(line);
        std::unordered_map<std::string, std::string> tags;
        for (std::string tag; std::getline(ss, tag, ' ');) {
            int index = tag.find(':');
            std::string key = tag.substr(0, index);
            std::string value = tag.substr(index + 1, tag.size());
            // todo: finish this
//            tags[key] = value;
        }

        if (tags.size() == 0) {
            continue;
        }

//        bool same_dataset = tags.find("dataset") == ;
    }

    datafile.close();
}

int main() {
    auto *geneticAlgorithm = new GeneticAlgorithm();
    geneticAlgorithm->run();
    delete geneticAlgorithm;
    return 0;
}
