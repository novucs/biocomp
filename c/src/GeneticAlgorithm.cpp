#include "GeneticAlgorithm.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <atomic>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "Individual.h"
#include "Random.h"

GeneticAlgorithm::GeneticAlgorithm(std::string dataset, std::vector<double> splits) : dataset_filename(dataset) {
    auto datasets = Dataset(dataset).split(splits);
    train = datasets.at(0);
    test = datasets.size() < 2 ? train : datasets.at(datasets.size() - 1);
    cross_validation = datasets.size() < 3 ? test : datasets.at(1);
    executor = new ThreadPool();

    int slash_position = dataset.rfind('/') + 1;
    int until_dot = dataset.rfind('.') - slash_position;
    dataset_name = dataset.substr(slash_position, until_dot);
    start_time = current_time();
    log_filename = "../logs/" + dataset_name + "/" + start_time + ".log";

    std::string create_log_directory_command = "mkdir -p ../logs/" + dataset_name;
    system(create_log_directory_command.c_str());

    std::ofstream file;
    file.open(log_filename, std::ios::app);
    file << "================================" << std::endl;
    file << "Run settings:" << std::endl;
    file << "\tdataset:" << dataset_filename << std::endl;
    file << "\trule_count:" << rule_count << std::endl;
    file << "\tpopulation_size:" << population_size << std::endl;
    file << "\tcrossover_chance:" << crossover_chance << std::endl;
    file << "\tmutation_rate:" << mutation_rate << std::endl;
    file << "\tselection_switch_threshold:" << selection_switch_threshold << std::endl;
    file << "\tcovered_best_variations:" << covered_best_variations << std::endl;
    file << "\ttournament_size:" << tournament_size << std::endl;
    file << "\tdistill_inheritance_chance:" << distill_inheritance_chance << std::endl;
    file << "\tcover_chance:" << cover_chance << std::endl;
    file << "\tfitness_threshold:" << fitness_threshold << std::endl;
    file << "================================" << std::endl;
    file.close();
}

bool GeneticAlgorithm::should_mutate() {
    return rng() < mutation_rate / (rule_count * train.features.at(0).size());
}

int GeneticAlgorithm::get_condition_size() {
    return train.features.at(0).size();
}

int GeneticAlgorithm::get_rule_count() {
    if (rule_count == 0) {
        rule_count = std::min((int) train.features.size(), 30);
    }
    return rule_count;
}

void GeneticAlgorithm::load_population(std::string filename) {
    std::ifstream datafile(filename);

    Individual best = generate_individual(this);
    int best_rule_count = std::numeric_limits<int>::max();
    double best_fitness = -1;

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

        bool same_dataset = tags["dataset"] == train.filename;
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
        population = generate_reduced_population(best);
    } else if (best_fitness == -1) {
        population = generate_covered_population();
    } else {
        population = generate_similar_population(best);
    }
}

std::vector<Individual> GeneticAlgorithm::generate_similar_population(Individual &best) {
    std::vector<Individual> target = generate_covered_population();
    target.at(0) = best;
    return target;
}

std::vector<Individual> GeneticAlgorithm::generate_reduced_population(Individual &best) {
    std::vector<Individual> target;
    for (int i = 0; i < population_size; i++) {
        if (rng() < distill_inheritance_chance) {
            target.push_back(best.remove_rule());
        } else {
            target.push_back(individual_from_samples(this, train));
        }
    }
    return target;
}

std::vector<Individual> GeneticAlgorithm::generate_covered_population() {
    std::vector<Individual> target;
    for (int i = 0; i < population_size; i++) {
        target.push_back(individual_from_samples(this, train));
    }
    return target;
}

Individual GeneticAlgorithm::tournament_selection(std::vector<double> &population_fitness) {
    Individual fittest = generate_individual(this);
    double fitness = -1;
    for (int i = 0; i < tournament_size; i++) {
        int index = rng() * population_size;
        if (population_fitness.at(index) > fitness) {
            fitness = population_fitness.at(index);
            fittest = population.at(index);
        }
    }
    return fittest;
}

Individual GeneticAlgorithm::create_offspring(std::vector<double> &population_fitness) {
    Individual offspring;

    if (rng() < crossover_chance) {
        Individual mum = tournament_selection(population_fitness);
        Individual dad = tournament_selection(population_fitness);
        offspring = mum.crossover(dad).mutate();
    } else {
        offspring = tournament_selection(population_fitness).mutate();
    }

    if (rng() < cover_chance) {
        offspring = offspring.cover(train);
    }

    return offspring;
}

void GeneticAlgorithm::run() {
    load_population("../solutions.txt");
    generation = 0;
    while (running) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            generation += 1;
            train_step();
            if (generation % 50 == 0) {
                display_test_results();
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

void GeneticAlgorithm::display_test_results() {
    printf("Generation: %*d ", 5, generation);
    printf("\tBest Fitness: %.3f ", test_fitness.best);
    printf("\tMean Fitness: %.3f ", test_fitness.mean);
    printf("\tRule Count: %*d ", 3, rule_count);
    printf("\t<--- Test Set\n");
}

void GeneticAlgorithm::train_step() {
    update_fitness();

    if (overall_best_fitness < train_fitness.best) {
        overall_best = Individual(train_fitness.best_individual);
        overall_best_fitness = train_fitness.best;
        if (fitness_threshold <= overall_best_fitness) {
            found_new_best();
            return;
        }
    }

    if (generation % 5 == 0) {
        printf("Generation: %*d ", 5, generation);
        printf("\tBest Fitness: %.3f ", train_fitness.best);
        printf("\tMean Fitness: %.3f ", train_fitness.mean);
        printf("\tRule Count: %*d ", 3, rule_count);
        printf("\n");
    }

    double mean_fitness_difference = std::max(0.0, train_fitness.mean - cross_validation_fitness.mean);

    std::vector<Individual> new_population;
    for (int i = 0; i < (population_size - covered_best_variations - 1); i++) {
        // todo: see if rule count could be used here
        // weigh selection fitness by that of the cross-validation set when over fitting
        if ((rng() * mean_fitness_difference) > selection_switch_threshold) {
            new_population.push_back(create_offspring(cross_validation_fitness.values));
        } else {
            new_population.push_back(create_offspring(train_fitness.values));
        }
    }

    new_population.push_back(Individual(overall_best));
    for (int i = 0; i < covered_best_variations; i++) {
        new_population.push_back(overall_best.cover(train).mutate());
    }

    population = new_population;
}

void GeneticAlgorithm::update_fitness() {
    std::atomic<int> updated(0);
    std::vector<FitnessAggregate> aggregates;
    std::vector<Dataset *> datasets = {&train, &cross_validation, &test};
    int splits = std::ceil(std::thread::hardware_concurrency() / (double) datasets.size());
    int expected_aggregates = datasets.size() * splits;
    aggregates.resize(expected_aggregates);

    for (int i = 0; i < datasets.size() * splits; i++) {
        executor->submit([this, &updated, &aggregates, &datasets, &splits, i] {
            int dataset_id = i / splits;
            int limit = (population.size() / (double) splits);
            int offset = limit * (i % splits);

            // scoop up remaining individuals in final batch
            if ((population.size() - offset - limit) < limit) {
                limit = population.size() - offset;
            }

            aggregates.at(i) = fitness_aggregate_of(*datasets.at(dataset_id), population, offset, limit);
            updated += 1;
        });
    }

    while (updated != expected_aggregates) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
    }

    train_fitness = combine_fitness_aggregates(
            std::vector<FitnessAggregate>(aggregates.begin() + (splits * 0), aggregates.begin() + (splits * 1)));
    cross_validation_fitness = combine_fitness_aggregates(
            std::vector<FitnessAggregate>(aggregates.begin() + (splits * 1), aggregates.begin() + (splits * 2)));
    test_fitness = combine_fitness_aggregates(
            std::vector<FitnessAggregate>(aggregates.begin() + (splits * 2), aggregates.begin() + (splits * 3)));

    log();
}

bool GeneticAlgorithm::found_new_best() {
    overall_best = overall_best.compress();
    rule_count = overall_best.rule_count();
    printf("Found rule in %d generations with rule count of %d\n", generation, rule_count);
    save_solution();
    overall_best_fitness = -1;
    rule_count -= 1;
    population = generate_reduced_population(overall_best);
    return true;
}

void GeneticAlgorithm::save_solution() {
    std::ofstream file;
    file.open("../solutions.txt", std::ios::app);
    file << "dataset:" << dataset_filename << " ";
    file << "rule_count:" << overall_best.rule_count() << " ";
    file << "generation:" << generation << " ";
    file << "fitness:" << test_fitness.best << " ";
    file << "time:" << current_time() << " ";
    file << "rules:" << overall_best.dump() << std::endl;
    file.close();
}

void GeneticAlgorithm::log() {
    std::ofstream file;
    file.open(log_filename, std::ios::app);
    file << "rule_count:" << rule_count << " ";
    file << "generation:" << generation << " ";
    file << "time:" << current_time() << " ";

    file << "train_fitness_best:" << train_fitness.best << " ";
    file << "train_fitness_mean:" << train_fitness.mean << " ";
    file << "train_fitness_first_quartile:" << train_fitness.first_quartile << " ";
    file << "train_fitness_median:" << train_fitness.median << " ";
    file << "train_fitness_third_quartile:" << train_fitness.third_quartile << " ";

    file << "cross_validation_fitness_best:" << cross_validation_fitness.best << " ";
    file << "cross_validation_fitness_mean:" << cross_validation_fitness.mean << " ";
    file << "cross_validation_fitness_first_quartile:" << cross_validation_fitness.first_quartile << " ";
    file << "cross_validation_fitness_median:" << cross_validation_fitness.median << " ";
    file << "cross_validation_fitness_third_quartile:" << cross_validation_fitness.third_quartile << " ";

    file << "test_fitness_best:" << test_fitness.best << " ";
    file << "test_fitness_mean:" << test_fitness.mean << " ";
    file << "test_fitness_first_quartile:" << test_fitness.first_quartile << " ";
    file << "test_fitness_median:" << test_fitness.median << " ";
    file << "test_fitness_third_quartile:" << test_fitness.third_quartile << " ";

    file << "best_train_rules:" << overall_best.dump() << " ";
    file << "best_cross_validation_rules:" << cross_validation_fitness.best_individual.dump() << " ";
    file << "best_test_rules:" << test_fitness.best_individual.dump() << std::endl;
    file.close();
}

std::string GeneticAlgorithm::current_time() {
    time_t now = time(0);
    struct tm *tstruct = localtime(&now);
    char current_time[80];
    strftime(current_time, sizeof(current_time), "%Y-%m-%d.%X", tstruct);
    return std::string(current_time);
}

void GeneticAlgorithm::terminate() {
    std::unique_lock<std::mutex> lock(mutex);
    running = false;
    std::string create_solutions_directory_cmd = "mkdir -p ../solutions/" + dataset_name;
    system(create_solutions_directory_cmd.c_str());
    std::string solutions_file = "../solutions/" + dataset_name + "/" + start_time + ".log";
    std::string copy_solutions_cmd = "cp ../solutions.txt " + solutions_file;
    system(copy_solutions_cmd.c_str());
    std::cout << "Saved solutions in " << solutions_file << std::endl;
}
