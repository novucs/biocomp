#include <sstream>
#include "Bounds.h"
#include "GeneticAlgorithm.h"
#include "Random.h"

double UPPER_LIMIT = 1.25;
double LOWER_LIMIT = -0.25;

Bounds::Bounds(GeneticAlgorithm *ga, double lower, double upper, bool wildcard) : ga(ga), lower(lower), upper(upper),
                                                                                  wildcard(wildcard) {}

bool Bounds::is_wildcard() {
    return wildcard;
}

Bounds Bounds::mutate() {
    if (is_wildcard()) {
        return rng() < ga->mutation_chance() ? random_bounds(ga) : *this;
    }

    if (rng() < ga->mutation_chance()) {
        return Bounds(ga, LOWER_LIMIT, UPPER_LIMIT, true);
    }

    auto new_lower = rng() < ga->mutation_chance() ? ((upper * rng()) + LOWER_LIMIT) : lower;
    auto new_upper = rng() < ga->mutation_chance() ? ((UPPER_LIMIT * rng()) + new_lower) : upper;
    return Bounds(ga, new_lower, new_upper, false);
}

bool Bounds::contains(double feature) {
    if (wildcard) {
        return true;
    }
    return lower < feature && feature < upper;
}

std::string Bounds::dump() {
    if (wildcard) {
        return "#";
    }
    return std::to_string(lower) + "~" + std::to_string(upper);
}

bool Bounds::subsumes(Bounds &other) {
    if (wildcard) {
        return true;
    }
    if (other.wildcard) {
        return false;
    }
    return lower <= other.lower && other.upper <= upper;
}

Bounds random_bounds(GeneticAlgorithm *ga, double surrounding) {
    double lower = uniform(LOWER_LIMIT, surrounding);
    double upper = uniform(surrounding, UPPER_LIMIT);
    return Bounds(ga, lower, upper, false);
}

Bounds random_bounds(GeneticAlgorithm *ga) {
    double lower = uniform(LOWER_LIMIT, UPPER_LIMIT);
    double upper = uniform(lower, UPPER_LIMIT);
    return Bounds(ga, lower, upper, false);
}

Bounds load_bounds(GeneticAlgorithm *ga, std::string dump) {
    if (dump == "#") {
        return Bounds(ga, LOWER_LIMIT, UPPER_LIMIT, true);
    }

    std::stringstream ss(dump);
    double upper, lower;
    char ignore;
    ss >> lower >> ignore >> upper;
    return Bounds(ga, lower, upper, false);
}
