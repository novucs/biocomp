#ifndef C_GENETICALGORITHM_H
#define C_GENETICALGORITHM_H

#include <vector>
#include "Dataset.h"
#include "Individual.h"
#include "FitnessAggregate.h"
#include "ThreadPool.h"

class GeneticAlgorithm {
private:
    std::string dataset_name;
    std::string dataset_filename;
    std::string log_filename;
    std::string start_time;
    Dataset train = Dataset();
    Dataset cross_validation = Dataset();
    Dataset test = Dataset();
    int rule_count = 60;
    int population_size = 100;
    double crossover_chance = 0.85;
    double mutation_rate = 0.003;
    bool use_tournament_selection = false;  // false uses roulette wheel selection
    double selection_switch_threshold = 0.1;
    int covered_best_variations = 5;
    int tournament_size = 5;
    double distill_inheritance_chance = 0.33;
    std::vector<Individual> population = std::vector<Individual>();
    Individual overall_best = dummy_individual();
    FitnessAggregate train_fitness = FitnessAggregate();
    FitnessAggregate cross_validation_fitness = FitnessAggregate();
    FitnessAggregate test_fitness = FitnessAggregate();
    double overall_best_fitness = -1;
    int generation = 0;
    double cover_chance = 0.1;
    double fitness_threshold = 60 / (double) 60;

    ThreadPool *executor;
    std::mutex mutex;
    bool running = true;
    bool load_population_from_file = false;
    int max_generation_count = 500;

public:
    GeneticAlgorithm(std::string dataset, std::vector<double> splits);

    bool should_mutate();

    int get_condition_size();

    int get_rule_count();

    void load_population(std::string filename);

    std::vector<Individual> generate_similar_population(Individual &best);

    std::vector<Individual> generate_reduced_population(Individual &best);

    std::vector<Individual> generate_covered_population();

    Individual roulette_wheel_selection(FitnessAggregate &fitness_aggregate);

    Individual tournament_selection(FitnessAggregate &fitness_aggregate);

    Individual select_parent(FitnessAggregate &fitness_aggregate);

    Individual create_offspring(FitnessAggregate &fitness_aggregate);

    void run();

    void display_test_results();

    void train_step();

    bool found_new_best();

    void save_solution();

    void update_fitness();

    std::string current_time();

    void log();

    void terminate();
};

#endif //C_GENETICALGORITHM_H
