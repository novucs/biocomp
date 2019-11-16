#include "GeneticAlgorithm.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "Individual.h"
#include "Random.h"

GeneticAlgorithm::GeneticAlgorithm(std::string dataset, std::vector<double> splits) {
    this->dataset = dataset;
    auto *datasets = Dataset(dataset).split(splits);
    this->train = datasets->at(0);
    this->test = datasets->size() < 2 ? this->train : datasets->at(datasets->size() - 1);
    this->cross_validation = datasets->size() < 3 ? this->test : datasets->at(1);
    delete datasets;
}

GeneticAlgorithm::~GeneticAlgorithm() {
    delete train;
    delete cross_validation;
    delete test;
}

double GeneticAlgorithm::mutation_chance() {
    return 0.00125;
}

int GeneticAlgorithm::get_condition_size() {
    return train->features->at(0).size();
}

int GeneticAlgorithm::get_rule_count() {
    if (rule_count == 0) {
        rule_count = std::min((int) train->features->size(), 30);
    }
    return rule_count;
}

void GeneticAlgorithm::set_population(std::vector<Individual *> *new_population) {
    if (population != nullptr) {
        for (Individual *individual : *population) {
            delete individual;
        }
    }
    delete population;
    population = new_population;
}

void GeneticAlgorithm::load_population(std::string filename) {
    std::ifstream datafile(filename);

    Individual *best = nullptr;
    int best_rule_count = std::numeric_limits<int>::max();
    double best_fitness = 0;

    for (std::string line; getline(datafile, line);) {
        std::stringstream ss(line);
        std::unordered_map<std::string, std::string> tags;

        for (std::string tag; std::getline(ss, tag, ' ');) {
            int index = tag.find(':');
            std::string key = tag.substr(0, index);
            std::string value = tag.substr(index + 1, tag.size());
            tags[key] = value;
        }

        if (tags.size() == 0) {
            continue;
        }

        bool same_dataset = tags["dataset"] == train->filename;
        int solution_rule_count = std::stoi(tags["rule_count"]);
        bool good_fitness = std::stod(tags["fitness"]) >= fitness_threshold;

        if (same_dataset && solution_rule_count < best_rule_count && good_fitness) {
            rule_count = best_rule_count = solution_rule_count;
            best = load_individual(this, tags["rules"]);
            best_fitness = std::stod(tags["fitness"]);
        }
    }

    datafile.close();

    bool reduce_rule_count = best_fitness >= fitness_threshold;
    if (reduce_rule_count) {
        rule_count -= 1;
    }

    generate_population(best, reduce_rule_count);
    delete best;
}

void GeneticAlgorithm::generate_population(Individual *best, bool smaller) {
    std::vector<Individual *> *new_population;
    if (best == nullptr) {
        new_population = generate_covered_population();
    } else if (smaller) {
        new_population = generate_reduced_population(best);
    } else {
        new_population = generate_similar_population(best);
    }
    set_population(new_population);
}

std::vector<Individual *> *GeneticAlgorithm::generate_similar_population(Individual *best) {
    std::vector<Individual *> *target = generate_covered_population();
    delete target->at(0);
    target->at(0) = best->copy();
    return target;
}

std::vector<Individual *> *GeneticAlgorithm::generate_reduced_population(Individual *best) {
    auto *target = new std::vector<Individual *>();
    for (int i = 0; i < population_size; i++) {
        if (rng() < distill_inheritance_chance) {
            target->push_back(best->remove_rule());
        } else {
            target->push_back(individual_from_samples(this, train->features, train->labels));
        }
    }
    return target;
}

std::vector<Individual *> *GeneticAlgorithm::generate_covered_population() {
    auto *target = new std::vector<Individual *>();
    for (int i = 0; i < population_size; i++) {
        target->push_back(individual_from_samples(this, train->features, train->labels));
    }
    return target;
}

Individual *GeneticAlgorithm::tournament_selection(std::vector<double> *population_fitness) {
    Individual *fittest = nullptr;
    double fitness = -1;
    for (int i = 0; i < tournament_size; i++) {
        int index = rng() * population_size;
        if (population_fitness->at(index) > fitness) {
            fitness = population_fitness->at(index);
            fittest = population->at(index);
        }
    }
    return fittest;
}

Individual *GeneticAlgorithm::create_offspring(std::vector<double> *population_fitness) {
    Individual *mum = tournament_selection(population_fitness);
    Individual *dad = tournament_selection(population_fitness);
    Individual *crossed = mum->crossover(dad);
    Individual *mutated = crossed->mutate();
    Individual *offspring = mutated;
    delete crossed;

    if (rng() < cover_chance) {
        offspring = offspring->cover(train->features, train->labels);
        delete mutated;
    }

    return offspring;
}

void GeneticAlgorithm::run() {
    load_population("../solutions.txt");
    generation = 0;
    while (running) {
        generation += 1;
        train_step();
        if (generation % 50 == 0) {
            display_test_results();
        }
    }
}

void GeneticAlgorithm::display_test_results() {
    std::vector<double> population_fitness;
    Individual *best_individual = nullptr;
    double best_fitness = -1;
    double total_fitness = 0;

    for (Individual *individual : *population) {
        double fitness = individual->fitness(test->features, test->labels);
        population_fitness.push_back(fitness);
        total_fitness += fitness;

        if (best_fitness < fitness) {
            best_individual = individual;
            best_fitness = fitness;
        }
    }

    double mean_fitness = total_fitness / population_fitness.size();
    printf("Generation: %*d ", 5, generation);
    printf("\tBest Fitness: %.3f ", best_fitness);
    printf("\tMean Fitness: %.3f ", mean_fitness);
    printf("\tRule Count: %*d ", 3, rule_count);
    printf("\t<--- Test Set\n");
}

void GeneticAlgorithm::train_step() {
    auto *train_fitness = new std::vector<double>();
    Individual *best_individual = nullptr;
    double best_fitness = -1;
    double total_train_fitness = 0;

    for (Individual *individual : *population) {
        double fitness = individual->fitness(train->features, train->labels);
        train_fitness->push_back(fitness);
        total_train_fitness += fitness;

        if (best_fitness < fitness) {
            best_individual = individual;
            best_fitness = fitness;
        }
    }

    // do not continue to process when best individual not found
    if (best_individual == nullptr) {
        delete train_fitness;
        return;
    }

    double mean_fitness = total_train_fitness / train_fitness->size();

    if (overall_best_fitness < best_fitness) {
        if (overall_best != nullptr) {
            delete overall_best;
        }
        overall_best = best_individual->copy();
        overall_best_fitness = best_fitness;

        bool regenerated_population = found_new_best(best_individual, best_fitness);
        if (regenerated_population) {
            delete train_fitness;
            return;
        }
    }

    if (generation % 5 == 0) {
        printf("Generation: %*d ", 5, generation);
        printf("\tBest Fitness: %.3f ", best_fitness);
        printf("\tMean Fitness: %.3f ", mean_fitness);
        printf("\tRule Count: %*d ", 3, rule_count);
        printf("\n");
    }

    double total_cross_validation_fitness = 0;
    auto *cross_validation_fitness = new std::vector<double>();
    for (Individual *individual : *population) {
        double fitness = individual->fitness(cross_validation->features, cross_validation->labels);
        cross_validation_fitness->push_back(fitness);
        total_cross_validation_fitness += fitness;
    }
    double cross_validation_mean_fitness = total_cross_validation_fitness / cross_validation->features->size();
    double mean_fitness_difference = std::max(0.0, mean_fitness - cross_validation_mean_fitness);

    auto *new_population = new std::vector<Individual *>();
    for (int i = 0; i < (population_size - 1); i++) {
        // todo: see if rule count could be used here
        // weigh selection fitness by that of the cross-validation set when over fitting
        if ((rng() * mean_fitness_difference) < selection_switch_threshold) {
            new_population->push_back(create_offspring(cross_validation_fitness));
        } else {
            new_population->push_back(create_offspring(train_fitness));
        }
    }
    new_population->push_back(overall_best->copy());
    set_population(new_population);
    delete cross_validation_fitness;
    delete train_fitness;
}

bool GeneticAlgorithm::found_new_best(Individual *best_individual, double best_fitness) {
    if (best_fitness < fitness_threshold) {
        delete overall_best;
        overall_best = best_individual->copy();
        overall_best_fitness = best_fitness;
        return false;
    }

    Individual *compressed = best_individual->compress();
    rule_count = compressed->rule_count();
    printf("Found rule in %d generations with rule count of %d\n", generation, rule_count);
    save_solution(best_individual, best_fitness, "../solutions.txt");

    delete overall_best;
    overall_best = nullptr;
    overall_best_fitness = -1;
    rule_count -= 1;
    generate_population(compressed, true);
    delete compressed;
    return true;
}

void GeneticAlgorithm::save_solution(Individual *best, double test_fitness, std::string filename) {
    if (test_fitness < fitness_threshold && !checkpoint_fitness) {
        return;
    }

    time_t now = time(0);
    struct tm tstruct = *localtime(&now);
    char current_time[80];
    strftime(current_time, sizeof(current_time), "%Y-%m-%d.%X", &tstruct);

    std::ofstream file;
    file.open(filename, std::ios::app);
    file << "dataset:" << dataset << " ";
    file << "rule_count:" << best->rule_count() << " ";
    file << "generation:" << generation << " ";
    file << "fitness:" << test_fitness << " ";
    file << "time:" << current_time << " ";
    file << "rules:" << best->dump() << std::endl;
    file.close();
}
