#include "FitnessAggregate.h"
#include <algorithm>
#include <tuple>

FitnessAggregate::FitnessAggregate(
        std::vector<double> &values,
        std::vector<double> &sorted_values,
        double total,
        double mean,
        double first_quartile,
        double median,
        double third_quartile,
        double best,
        Individual &best_individual
)
        : values(values),
          sorted_values(sorted_values),
          total(total),
          mean(mean),
          first_quartile(first_quartile),
          median(median),
          third_quartile(third_quartile),
          best(best),
          best_individual(best_individual) {}

FitnessAggregate::FitnessAggregate() {}

std::tuple<double, double, double> quartiles(std::vector<double> sorted_values) {
    double first_quartile = sorted_values.at(sorted_values.size() * 0.25);
    double median = sorted_values.at(sorted_values.size() * 0.5);
    double third_quartile = sorted_values.at(sorted_values.size() * 0.75);
    return {first_quartile, median, third_quartile};
}

FitnessAggregate combine_fitness_aggregates(std::vector<FitnessAggregate> aggregates) {
    std::vector<double> values;
    values.reserve(aggregates.size() * aggregates.at(0).values.size());

    double total = 0;
    double best = -1;
    Individual best_individual;

    for (FitnessAggregate &aggregate : aggregates) {
        values.insert(values.end(), aggregate.values.begin(), aggregate.values.end());
        total += aggregate.total;

        if (best < aggregate.best) {
            best = aggregate.best;
            best_individual = aggregate.best_individual;
        }
    }

    double mean = total / values.size();

    std::vector<double> sorted_values(values);
    std::stable_sort(sorted_values.begin(), sorted_values.end());
    auto[first_quartile, median, third_quartile] = quartiles(sorted_values);

    return FitnessAggregate(
            values,
            sorted_values,
            total,
            mean,
            first_quartile,
            median,
            third_quartile,
            best,
            best_individual
    );
}

FitnessAggregate fitness_aggregate_of(Dataset &dataset, std::vector<Individual> &population, int offset, int limit) {
    std::vector<double> values;
    Individual best_individual;
    double best = -1;
    double total = 0;

    for (int i = offset; i < (offset + limit); i++) {
        Individual &individual = population.at(i);
        double fitness = individual.fitness(dataset);
        values.push_back(fitness);
        total += fitness;

        if (best < fitness) {
            best = fitness;
            best_individual = individual;
        }
    }

    double mean = total / (double) limit;

    std::vector<double> sorted_values(values);
    std::stable_sort(sorted_values.begin(), sorted_values.end());

    auto[first_quartile, median, third_quartile] = quartiles(sorted_values);
    return FitnessAggregate(
            values,
            sorted_values,
            total,
            mean,
            first_quartile,
            median,
            third_quartile,
            best,
            best_individual
    );
}
