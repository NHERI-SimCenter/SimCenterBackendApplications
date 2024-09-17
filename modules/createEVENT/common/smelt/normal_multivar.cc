#include <iostream>
#include <stdexcept>
// Boost random generator
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
// Eigen dense matrices
#include <Eigen/Dense>

#include "factory.h"
#include "normal_multivar.h"

namespace numeric_utils {

NormalMultiVar::NormalMultiVar()
  : RandomGenerator()
{
  generator_ = boost::random::mt19937(seed_);
  distribution_ = boost::random::normal_distribution<double>();
}

NormalMultiVar::NormalMultiVar(int seed)
  : RandomGenerator()
{
  seed_ = seed;
  generator_ = boost::random::mt19937(seed_);
  distribution_ = boost::random::normal_distribution<double>();
}

bool NormalMultiVar::generate(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& random_numbers,
    const Eigen::VectorXd& means, const Eigen::MatrixXd& cov,
    unsigned int cases) {

  bool success = true;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lower_cholesky;
  
  try {
    auto llt = cov.llt();
    lower_cholesky = llt.matrixL();

    if (llt.info() == Eigen::NumericalIssue) {
      throw std::runtime_error(
          "\nERROR: In NormalMultivar::generate method: Input covariance matrix is not "
          "positive semi-definite\n");
    }
  } catch (const std::exception& e) {
    std::cerr << "\nERROR: In normal multivariate random number generation: "
              << e.what() << std::endl;
    success = false;
  }

  random_numbers.resize(cov.rows(), cases);

  // Generate random numbers based on distribution and generator type for
  // requested number of cases
  for (unsigned int i = 0; i < random_numbers.cols(); ++i) {
    for (unsigned int j = 0; j < random_numbers.rows(); ++j) {
      random_numbers(j, i) = distribution_(generator_);
    }
  }

  // Transform from unit normal distribution based on covariance and mean values
  for (unsigned int i = 0; i < random_numbers.cols(); ++i) {
    random_numbers.col(i) = lower_cholesky * random_numbers.col(i) + means;
  }

  return success;
}

std::string NormalMultiVar::name() const {
  return "NormalMultiVar";
}
}  // namespace numeric_utils
