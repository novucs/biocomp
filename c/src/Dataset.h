#ifndef C_DATASET_H
#define C_DATASET_H

#include <vector>
#include <string>

class Dataset {
private:
    void load_floating_point_features(const std::string &line, std::stringstream &ss);

    void load_binary_features(std::stringstream &ss);

public:
    std::string filename;
    std::vector<std::vector<double>> features;
    std::vector<int> labels;

    Dataset();

    Dataset(std::string filename);

    Dataset(std::string filename, std::vector<std::vector<double>> features, std::vector<int> labels);

    std::vector<Dataset> split(std::vector<double> ways);
};

#endif //C_DATASET_H
