#include <vector>
#include <boost/math/distributions/beta.hpp>
#include "beta_dist.h"

stochastic::BetaDistribution::BetaDistribution(double alpha, double beta)
  : Distribution(),
    alpha_{alpha},
    beta_{beta},
    distribution_{alpha, beta_}
{}

std::vector<double> stochastic::BetaDistribution::cumulative_dist_func(
    const std::vector<double>& locations) const {
  std::vector<double> evaluations(locations.size());

  for (unsigned int i = 0; i < locations.size(); ++i) {
    evaluations[i] = cdf(distribution_, locations[i]);
  }

  return evaluations;
}

std::vector<double> stochastic::BetaDistribution::inv_cumulative_dist_func(
    const std::vector<double>& probabilities) const {
  std::vector<double> evaluations(probabilities.size());

  for (unsigned int i = 0; i < probabilities.size(); ++i) {
    evaluations[i] = quantile(distribution_, probabilities[i]);
  }

  return evaluations;
}
