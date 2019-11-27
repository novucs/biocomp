#include "Main.h"
#include <iostream>
#include "GeneticAlgorithm.h"

int main(int argc, char *argv[]) {
    if (argc <= 2 || argc == 4) {
        std::cout << "Expected usage:" << std::endl;
        std::cout << "\t./rulebasedga {dataset} {train%} [cross_validation%] [test%]" << std::endl;
        std::cout << "\te.g.: ./rulebasedga ../../datasets/2019/data1.txt 1.0" << std::endl;
        std::cout << "\t      ./rulebasedga ../../datasets/2019/data3.txt 0.3 0.3 1.0" << std::endl;
        return 0;
    }

    std::string dataset(argv[1]);
    double train = std::stod(argv[2]);
    std::vector<double> splits = {train};

    if (argc == 5) {
        double cross_validation = std::stod(argv[3]);
        double test = std::stod(argv[4]);
        splits = {train, cross_validation, test};
    }

    for (int i = 0; i < 5; i++) {
        GeneticAlgorithm ga = GeneticAlgorithm(dataset, splits, 0.0, 0.0001);
        ga.run();
    }

    return 0;
}
