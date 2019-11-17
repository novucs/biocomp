#include "Main.h"
#include "GeneticAlgorithm.h"

int main(int argc, char *argv[]) {
//    GeneticAlgorithm ga("../../datasets/2019/data1.txt", {1.0});
//    GeneticAlgorithm ga("../../datasets/2019/data2.txt", {1.0});
//    GeneticAlgorithm ga("../../datasets/2019/data3.txt", {0.3, 0.3, 1.0});
    GeneticAlgorithm ga("../../datasets/2019/data4.txt", {0.3, 0.3, 1.0});
    ga.run();
    return 0;
}
