#include <cstdlib>
#include <algorithm>
#include <sstream>
#include "Bounds.h"
#include "GeneticAlgorithm.h"

double UPPER_LIMIT = 1.25;
double LOWER_LIMIT = -0.25;

Bounds::Bounds(GeneticAlgorithm *ga, double upper, double lower) {
    this->ga = ga;
    this->upper = upper;
    this->lower = lower;
    this->general = false;
}

Bounds::Bounds(GeneticAlgorithm *ga) {
    this->ga = ga;
    this->upper = UPPER_LIMIT;
    this->lower = LOWER_LIMIT;
    this->general = true;
}

bool Bounds::is_generalisable() {
    return general;
}

Bounds *Bounds::mutate() {
    if (is_generalisable()) {
        return std::rand() < ga->mutation_chance() ? random_bounds(ga) : this;
    }

    if (std::rand() < ga->mutation_chance()) {
        return new Bounds(ga);
    }

    auto new_lower = std::rand() < ga->mutation_chance() ? ((upper * std::rand()) + LOWER_LIMIT) : lower;
    auto new_upper = std::rand() < ga->mutation_chance() ? ((UPPER_LIMIT * std::rand()) + new_lower) : upper;
    return new Bounds(ga, new_upper, new_lower);
}

bool Bounds::contains(double feature) {
    if (general) {
        return true;
    }
    return lower < feature && feature < upper;
}

std::string Bounds::dump() {
    if (general) {
        return "#";
    }
    return std::to_string(lower) + "~" + std::to_string(upper);
}

bool Bounds::subsumes(Bounds other) {
    if (general) {
        return true;
    }
    if (other.general) {
        return false;
    }
    return lower <= other.lower && other.upper <= upper;
}

Bounds *random_bounds(GeneticAlgorithm *ga, double surrounding) {
    double lower = (surrounding * std::rand()) + LOWER_LIMIT;
    double upper = (UPPER_LIMIT * std::rand()) + surrounding;
    return new Bounds(ga, upper, lower);
}

Bounds *random_bounds(GeneticAlgorithm *ga) {
    double lower = (UPPER_LIMIT * std::rand()) + LOWER_LIMIT;
    double upper = ((UPPER_LIMIT - lower) * std::rand()) + lower;
    return new Bounds(ga, upper, lower);
}

Bounds *load_bounds(GeneticAlgorithm *ga, std::string dump) {
    if (dump == "#") {
        return new Bounds(ga);
    }

    std::stringstream ss(dump);
    double upper, lower;
    char ignore;
    ss >> lower >> ignore >> upper;
    return new Bounds(ga, upper, lower);
}
