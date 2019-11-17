#include <algorithm>
#include "FitnessAggregate.h"

FitnessAggregate::FitnessAggregate(
        std::vector<double> &values,
        double total,
        double mean,
        double first_quartile,
        double median,
        double third_quartile,
        double best,
        Individual &best_individual
)
        : values(values),
          total(total),
          mean(mean),
          first_quartile(first_quartile),
          median(median),
          third_quartile(third_quartile),
          best(best),
          best_individual(best_individual) {}

FitnessAggregate::FitnessAggregate() {}

FitnessAggregate fitness_aggregate_of(Dataset dataset, std::vector<Individual> population) {
    std::vector<double> values;
    Individual best_individual;
    double best = -1;
    double total = 0;

    for (Individual &individual : population) {
        double fitness = individual.fitness(dataset);
        values.push_back(fitness);
        total += fitness;

        if (best < fitness) {
            best = fitness;
            best_individual = individual;
        }
    }

    double mean = total / (double) values.size();

    std::vector<double> sorted_values(values);
    std::stable_sort(sorted_values.begin(), sorted_values.end());
    double first_quartile = sorted_values.at(sorted_values.size() * 0.25);
    double median = sorted_values.at(sorted_values.size() * 0.5);
    double third_quartile = sorted_values.at(sorted_values.size() * 0.75);
    return FitnessAggregate(values, total, mean, first_quartile, median, third_quartile, best, best_individual);
}
