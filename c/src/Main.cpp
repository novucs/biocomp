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
    double cross_validation = 1.0;
    double test = 1.0;

    if (argc == 5) {
        cross_validation = std::stod(argv[3]);
        test = std::stod(argv[4]);
    }

    GeneticAlgorithm ga =
            argc == 5 ? GeneticAlgorithm(dataset, {train, cross_validation, test}) : GeneticAlgorithm(dataset, {train});
    std::thread ga_thread(&GeneticAlgorithm::run, &ga);

    std::string throwaway;
    std::getline(std::cin, throwaway);
    ga.terminate();
    ga_thread.join();
    return 0;
}
