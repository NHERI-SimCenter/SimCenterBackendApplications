#include <vector>
#include <boost/math/distributions/students_t.hpp>
#include "students_t_dist.h"

stochastic::StudentstDistribution::StudentstDistribution(double mean,
                                                         double std_dev,
                                                         double dof)
    : Distribution(),
      mean_{mean},
      std_dev_{std_dev},
      dof_{dof},
      distribution_{dof_}
{}

std::vector<double> stochastic::StudentstDistribution::cumulative_dist_func(
    const std::vector<double>& locations) const {
  std::vector<double> evaluations(locations.size());

  for (unsigned int i = 0; i < locations.size(); ++i) {
    evaluations[i] = cdf(distribution_, (locations[i] - mean_) / std_dev_);
  }

  return evaluations;
}

std::vector<double> stochastic::StudentstDistribution::inv_cumulative_dist_func(
    const std::vector<double>& probabilities) const {
  std::vector<double> evaluations(probabilities.size());

  for (unsigned int i = 0; i < probabilities.size(); ++i) {
    evaluations[i] =
        std_dev_ * quantile(distribution_, probabilities[i]) + mean_;
  }

  return evaluations;
}
