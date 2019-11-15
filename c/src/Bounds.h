#ifndef C_BOUNDS_H
#define C_BOUNDS_H

#include <string>

class GeneticAlgorithm;

class Bounds {
private:
    GeneticAlgorithm *ga;
    double lower;
    double upper;
    bool general;
public:
    Bounds(GeneticAlgorithm *ga, double upper, double lower);

    explicit Bounds(GeneticAlgorithm *ga);

    bool is_generalisable();

    Bounds *mutate();

    bool contains(double feature);

    std::string dump();

    bool subsumes(Bounds other);
};

Bounds *random_bounds(GeneticAlgorithm *ga, double surrounding);

Bounds *random_bounds(GeneticAlgorithm *ga);

Bounds *load_bounds(GeneticAlgorithm *ga, std::string dump);

#endif //C_BOUNDS_H
