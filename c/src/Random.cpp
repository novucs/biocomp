#include <random>
#include "Random.h"


std::random_device rd;
std::default_random_engine e2(rd());
std::uniform_real_distribution<> distribution(0.0, 1.0);

double rng() {
    return distribution(e2);
}

double uniform(double lower, double upper) {
    std::uniform_real_distribution<> dist(lower, upper);
    return dist(e2);
}
