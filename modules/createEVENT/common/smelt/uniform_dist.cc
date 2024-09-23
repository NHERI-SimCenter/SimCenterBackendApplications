#include <vector>
#include <boost/math/distributions/uniform.hpp>
#include "uniform_dist.h"

stochastic::UniformDistribution::UniformDistribution(double lower, double upper)
  : Distribution(),
    lower_bound_{lower},
    upper_bound_{upper},
    distribution_{lower, upper}
{}

std::vector<double> stochastic::UniformDistribution::cumulative_dist_func(
    const std::vector<double>& locations) const {
  std::vector<double> evaluations(locations.size());

  for (unsigned int i = 0; i < locations.size(); ++i) {
    evaluations[i] = cdf(distribution_, locations[i]);
  }

  return evaluations;
}

std::vector<double> stochastic::UniformDistribution::inv_cumulative_dist_func(
    const std::vector<double>& probabilities) const {
  std::vector<double> evaluations(probabilities.size());

  for (unsigned int i = 0; i < probabilities.size(); ++i) {
    evaluations[i] = quantile(distribution_, probabilities[i]);
  }

  return evaluations;
}
