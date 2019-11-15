#ifndef C_GENETICALGORITHM_H
#define C_GENETICALGORITHM_H

#include <vector>
#include "Dataset.h"

class Individual;

class GeneticAlgorithm {
private:
    std::string dataset;
    Dataset *train;
    Dataset *cross_validation;
    Dataset *test;
    int rule_count = 60;
    int population_size = 100;
    int generation_count = 1000;
    double crossover_chance = 0.5;
    int tournament_size = 5;
    double distill_inheritance_chance = 0.33;
    std::vector<Individual *> *population;
    Individual *overall_best = nullptr;
    double overall_best_fitness;
    int generation = 0;
    bool checkpoint_fitness = false;
    double cover_chance = 0.1;
    double fitness_threshold = 59 / (double) 60;
    bool verbose = false;
    bool running = true;

public:
    GeneticAlgorithm(std::string dataset);

    ~GeneticAlgorithm();

    double mutation_chance();

    int get_condition_size();

    int get_rule_count();

    void set_population(std::vector<Individual *> *new_population);

    void load_population(std::string filename);

    void generate_population(Individual *best, bool smaller);

    std::vector<Individual *> *generate_similar_population(Individual *best);

    std::vector<Individual *> *generate_reduced_population(Individual *best);

    std::vector<Individual *> *generate_covered_population();

    Individual *tournament_selection(std::vector<double> *population_fitness);

    Individual *create_offspring(std::vector<double> *population_fitness);

    void run();

    void display_test_results();

    void train_step();

    bool found_new_best(Individual *best_individual, double best_fitness);

    void save_solution(Individual *best, double test_fitness, std::string filename);
};

#endif //C_GENETICALGORITHM_H
