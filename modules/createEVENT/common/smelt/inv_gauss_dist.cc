#include <vector>
#include <boost/math/distributions/inverse_gaussian.hpp>
#include "inv_gauss_dist.h"

stochastic::InverseGaussianDistribution::InverseGaussianDistribution(
    double mean, double std_dev)
    : Distribution(),
      mean_{mean},
      std_dev_{std_dev},
      distribution_{mean, std_dev_}     
{}

std::vector<double>
    stochastic::InverseGaussianDistribution::cumulative_dist_func(
        const std::vector<double>& locations) const {
  std::vector<double> evaluations(locations.size());

  for (unsigned int i = 0; i < locations.size(); ++i) {
    evaluations[i] = cdf(distribution_, locations[i]);
  }

  return evaluations;
}

std::vector<double>
    stochastic::InverseGaussianDistribution::inv_cumulative_dist_func(
        const std::vector<double>& probabilities) const {
  std::vector<double> evaluations(probabilities.size());

  for (unsigned int i = 0; i < probabilities.size(); ++i) {
    evaluations[i] = quantile(distribution_, probabilities[i]);
  }

  return evaluations;
}
