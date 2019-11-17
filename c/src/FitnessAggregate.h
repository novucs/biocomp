#ifndef C_FITNESSAGGREGATE_H
#define C_FITNESSAGGREGATE_H

#include <vector>
#include "Individual.h"

class FitnessAggregate {
public:
    std::vector<double> values = std::vector<double>();
    std::vector<double> sorted_values = std::vector<double>();
    double total = 0;
    double mean = 0;
    double first_quartile = 0;
    double median = 0;
    double third_quartile = 0;
    double best = 0;
    Individual best_individual = dummy_individual();

    FitnessAggregate();

    FitnessAggregate(std::vector<double> &values,
                     std::vector<double> &sorted_values,
                     double total,
                     double mean,
                     double first_quartile,
                     double median,
                     double third_quartile,
                     double best,
                     Individual &best_individual);
};

FitnessAggregate combine_fitness_aggregates(std::vector<FitnessAggregate> aggregates);

FitnessAggregate fitness_aggregate_of(Dataset &dataset, std::vector<Individual> &population, int offset, int limit);

#endif //C_FITNESSAGGREGATE_H
