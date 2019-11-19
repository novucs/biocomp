#include <sstream>
#include "Bounds.h"
#include "GeneticAlgorithm.h"
#include "Random.h"

double UPPER_LIMIT = 1.25;
double LOWER_LIMIT = -0.25;

Bounds::Bounds(GeneticAlgorithm *ga, double lower, double upper, bool wildcard) : ga(ga), lower(lower), upper(upper),
                                                                                  wildcard(wildcard) {}

Bounds::Bounds(const Bounds &bounds) : ga(bounds.ga), lower(bounds.lower), upper(bounds.upper),
                                       wildcard(bounds.wildcard) {}

bool Bounds::is_wildcard() {
    return wildcard;
}

Bounds Bounds::mutate() {
    if (is_wildcard()) {
        return ga->should_mutate() ? random_bounds(ga) : *this;
    }

    if (ga->should_mutate()) {
        return Bounds(ga, LOWER_LIMIT, UPPER_LIMIT, true);
    }

    auto new_lower = ga->should_mutate() ? uniform(LOWER_LIMIT, upper) : lower;
    auto new_upper = ga->should_mutate() ? uniform(lower, UPPER_LIMIT) : upper;
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
    double bound1 = uniform(LOWER_LIMIT, UPPER_LIMIT);
    double bound2 = uniform(LOWER_LIMIT, UPPER_LIMIT);
    double lower = std::min(bound1, bound2);
    double upper = std::max(bound1, bound2);
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
