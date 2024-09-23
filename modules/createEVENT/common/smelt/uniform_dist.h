#ifndef _UNIFORM_DIST_H_
#define _UNIFORM_DIST_H_

#include <string>
#include <vector>
#include <boost/math/distributions/uniform.hpp>
#include "distribution.h"

namespace stochastic {
/**
 * Uniform distribution
 */
class UniformDistribution : public Distribution {
 public:
  /**
   * @constructor Construct standard uniform distribution
   */
  UniformDistribution() = delete;

  /**
   * @constructor Construct uniform distribution with specified lower and
   * upper bounds
   * @param[in] lower Lower bound
   * @param[in] upper Upper bound
   */
  UniformDistribution(double lower, double upper);

  /**
   * @destructor Virtual destructor
   */
  virtual ~UniformDistribution(){};

  /**
   * Get the name of the distribution model
   * @return Model name as a string
   */
  std::string name() const override { return "UniformDist"; };

  /**
   * Compute the cumulative distribution function (CDF) of the distribution at
   * specified input locations
   * @param[in] locations Vector containing locations at which to
   *                      calculate CDF
   * @return Vector of evaluated values of CDF at input locations
   */
  std::vector<double> cumulative_dist_func(
      const std::vector<double>& locations) const override;

  /**
   * Compute the inverse cumulative distribution function (ICDF) of the
   * distribution at specified input locations
   * @param[in] probabilities Vector containing probabilities at which to
   *                          calculate ICDF
   * @return Vector of evaluated values of ICDF at input locations
   */
  std::vector<double> inv_cumulative_dist_func(
      const std::vector<double>& probabilities) const override;

 protected:
  double lower_bound_;                /**< Distribution lower bound */
  double upper_bound_;                /**< Distribution upper bound */
  boost::math::uniform distribution_; /**< Uniform distribution */
};
}  // namespace stochastic

#endif  // _UNIFORM_DIST_H_
