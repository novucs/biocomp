#include "GeneticAlgorithm.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void GeneticAlgorithm::run() {
    std::cout << "Hello, World!" << std::endl;
    std::ifstream datafile("../../datasets/2019/data3.txt");
    for (std::string line; getline(datafile, line);) {
        getline(datafile, line);
        new std::vector<double>;
        std::cout << line << std::endl;
    }
    datafile.close();
}

int main() {
    auto *geneticAlgorithm = new GeneticAlgorithm();
    geneticAlgorithm->run();
    delete geneticAlgorithm;
    return 0;
}
