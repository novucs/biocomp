#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "Dataset.h"


Dataset::Dataset(std::string filename) {
    this->filename = filename;
    std::ifstream datafile(filename);

    this->features = new std::vector<std::vector<double>>();
    this->labels = new std::vector<int>();

    for (std::string line; getline(datafile, line);) {
        std::stringstream ss(line);

        if (line.find('.') != std::string::npos) {
            load_floating_point_features(line, ss);
        } else {
            load_binary_features(ss);
        }
    }

    datafile.close();
}

void Dataset::load_binary_features(std::stringstream &ss) const {
    std::string sample_features_string;
    std::getline(ss, sample_features_string, ' ');
    std::vector<double> sample_features;

    for (int i = 0; i < sample_features_string.size(); i++) {
        std::string feature(1, sample_features_string.at(i));
        sample_features.push_back(std::stod(feature));
    }

    int label;
    ss >> label;

    features->push_back(sample_features);
    labels->push_back(label);
}

void Dataset::load_floating_point_features(const std::string &line, std::stringstream &ss) const {
    // count number of spaces in the line
    int spaces = 0;
    for (char c : line) {
        if (c == ' ') {
            spaces += 1;
        }
    }

    // load all features
    std::vector<double> sample_features;
    for (int i = 0; i < spaces; i++) {
        double feature;
        ss >> feature;
        sample_features.push_back(feature);
    }

    // load label
    int label;
    ss >> label;

    // add to loaded dataset
    features->push_back(sample_features);
    labels->push_back(label);
}

Dataset::Dataset(std::string filename, std::vector<std::vector<double>> *features, std::vector<int> *labels) {
    this->filename = filename;
    this->features = features;
    this->labels = labels;
}

std::vector<Dataset> *Dataset::split(std::vector<double> ways) {
    std::vector<int> indices;
    for (int i = 0; i < features->size(); i++) {
        indices.push_back(i);
    }
    std::random_shuffle(indices.begin(), indices.end());

    auto *datasets = new std::vector<Dataset>();
    int index = 0;

    for (double percent : ways) {
        int target = std::min((int) features->size(), (int) (index + (features->size() * percent)));
        auto new_features = new std::vector<std::vector<double>>();
        auto new_labels = new std::vector<int>();
        for (; index < target; index++) {
            new_features->push_back(features->at(indices.at(index)));
            new_labels->push_back(labels->at(indices.at(index)));
        }
        Dataset dataset(filename, new_features, new_labels);
        datasets->push_back(dataset);
    }

    return datasets;
}
