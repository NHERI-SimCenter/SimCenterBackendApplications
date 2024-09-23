#ifndef _NUMERIC_UTILS_H_
#define _NUMERIC_UTILS_H_

#include <complex>
#include <ctime>
#include <utility>
#include <vector>
#include <Eigen/Dense>

/**
 * Numeric utility functions not tied to any particular class
 */
namespace numeric_utils {

/**
 * Convert input correlation matrix and standard deviation to covariance matrix
 * @param[in] corr Input correlation matrix
 * @param[in] std_dev Standard deviation vector
 * @return Covariance matrix with same dimensions as input correlation matrix
 */
Eigen::MatrixXd corr_to_cov(const Eigen::MatrixXd& corr,
			    const Eigen::VectorXd& std_dev);

/**
 * Compute the 1-dimensional convolution of two input vectors
 * @param[in] input_x First input vector of data
 * @param[in] input_y Second input vector of data
 * @param[out] output Vector to story convolution results to
 * @return Returns true if convolution was successful, false otherwise
 */
bool convolve_1d(const std::vector<double>& input_x,
                 const std::vector<double>& input_y,
                 std::vector<double>& response);

/**
 * Computes the real portion of the 1-dimensional inverse Fast Fourier Transform
 * (FFT) of the input vector
 * @param[in] input_vector Input vector to compute the inverse FFT of
 * @param[in, out] output_vector Vector to write output to
 * @return Returns true if computations were successful, false otherwise
 */
bool inverse_fft(std::vector<std::complex<double>> input_vector,
                 std::vector<double>& output_vector);

/**
 * Computes the real portion of the 1-dimensional inverse Fast Fourier Transform
 * (FFT) of the input vector
 * @param[in] input_vector Input vector to compute the inverse FFT of
 * @param[in, out] output_vector Vector to write output to
 * @return Returns true if computations were successful, false otherwise
 */ 
bool inverse_fft(const Eigen::VectorXcd& input_vector,
                 Eigen::VectorXd& output_vector);

/**
 * Computes the real portion of the 1-dimensional inverse Fast Fourier Transform
 * (FFT) of the input vector
 * @param[in] input_vector Input vector to compute the inverse FFT of
 * @param[in, out] output_vector Vector to write output to
 * @return Returns true if computations were successful, false otherwise
 */ 
bool inverse_fft(const Eigen::VectorXcd& input_vector,
                 std::vector<double>& output_vector);

/**
 * Computes the real portion of the 1-dimensional Fast Fourier Transform
 * (FFT) of the input vector
 * @param[in] input_vector Input vector to compute the FFT of
 * @param[in, out] output_vector Vector to write output to
 * @return Returns true if computations were successful, false otherwise
 */
bool fft(std::vector<double> input_vector,
         std::vector<std::complex<double>>& output_vector);

/**
 * Computes the real portion of the 1-dimensional Fast Fourier Transform
 * (FFT) of the input vector
 * @param[in] input_vector Input vector to compute the FFT of
 * @param[in, out] output_vector Vector to write output to
 * @return Returns true if computations were successful, false otherwise
 */
bool fft(const Eigen::VectorXd& input_vector, Eigen::VectorXcd& output_vector);

/**
 * Computes the real portion of the 1-dimensional Fast Fourier Transform
 * (FFT) of the input vector
 * @param[in] input_vector Input vector to compute the FFT of
 * @param[in, out] output_vector Vector to write output to
 * @return Returns true if computations were successful, false otherwise
 */
bool fft(const Eigen::VectorXd& input_vector,
         std::vector<std::complex<double>>& output_vector);

/**
 * Calculate the integral of the input vector with uniform spacing
 * between data points
 * @param[in] input_vector Vector containing function values
 * @param[in] spacing Spacing between data points
 * @return Approximate value of function integral
 */
double trapazoid_rule(const std::vector<double>& input_vector, double spacing);

/**
 * Calculate the integral of the input vector with uniform spacing
 * between data points
 * @param[in] input_vector Vector containing function values
 * @param[in] spacing Spacing between data points
 * @return Approximate value of function integral
 */
double trapazoid_rule(const Eigen::VectorXd& input_vector, double spacing);

/**
 * Fit polynomial to data, forcing y-intercept to zero
 * @param[in] points Vector of evaluation points
 * @param[in] data Vector of data for evaluation points
 * @param[in] intercept Value for y-intercept
 * @param[in] degree Degree of of polynomial fit
 */
Eigen::VectorXd polyfit_intercept(const Eigen::VectorXd& points,
                                  const Eigen::VectorXd& data,
				  double intercept,
                                  unsigned int degree);

/**
 * Take the derivative of a polynomial described by its coefficients
 * @param[in] coefficients Coefficients of polynomial terms ordered in
 *                         descending power
 * @return Vector of coefficients for input polynomial derivative
 */
Eigen::VectorXd polynomial_derivative(const Eigen::VectorXd& coefficients);

/**
 * Approximates the derivative as differences between adjacent input points
 * @param[in] coefficients Coefficients of polynomial terms ordered in
 *                         descending power
 * @param[in] constant_factor Constant factor to multiply coefficients by.
 *                            Defaults to 1.0.
 * @param[in] add_zero Boolean indicating to add leading zero to coefficients.
 *                     Defaults to false.
 * @return Vector of difference for input vector
 */
std::vector<double> derivative(
    const std::vector<double>& coefficients, double constant_factor = 1.0,
    bool add_zero = false);

/**
 * Evaluate polynomial described by input coefficients at input points
 * @param[in] coefficients Coefficients of polynomial terms ordered in
 *                         descending power
 * @param[in] points Vector of points at which to evaluate polynomial
 * @return Vector of polynomial values evaluated at input points
 */
Eigen::VectorXd evaluate_polynomial(const Eigen::VectorXd& coefficients,
                                    const Eigen::VectorXd& points);

/**
 * Evaluate polynomial described by input coefficients at input points
 * @param[in] coefficients Coefficients of polynomial terms ordered in
 *                         descending power
 * @param[in] points Vector of points at which to evaluate polynomial
 * @return Vector of polynomial values evaluated at input points
 */
Eigen::VectorXd evaluate_polynomial(const Eigen::VectorXd& coefficients,
                                    const std::vector<double>& points);

/**
 * Evaluate polynomial described by input coefficients at input points
 * @param[in] coefficients Coefficients of polynomial terms ordered in
 *                         descending power
 * @param[in] points Vector of points at which to evaluate polynomial
 * @return Vector of polynomial values evaluated at input points
 */
std::vector<double> evaluate_polynomial(const std::vector<double>& coefficients,
                                        const std::vector<double>& points);

/**
 * Abstract base class for random number generators
 */
class RandomGenerator {
 public:
  /**
   * @constructor Default constructor
   */
  RandomGenerator() = default;

  /**
   * @destructor Virtual destructor
   */
  virtual ~RandomGenerator() {};

  /**
   * Delete copy constructor
   */
  RandomGenerator(const RandomGenerator&) = delete;

  /**
   * Delete assignment operator
   */
  RandomGenerator& operator=(const RandomGenerator&) = delete;

  /**
   * Get multivariate random realization
   * @param[in, out] random_numbers Matrix to store generated random numbers to
   * @param[in] means Vector of mean values for random variables
   * @param[in] cov Covariance matrix of for random variables
   * @param[in] cases Number of cases to generate
   * @return Returns true if no issues were encountered in Cholesky
   *         decomposition of covariance matrix, returns false otherwise
   */
  virtual bool generate(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& random_numbers,
      const Eigen::VectorXd& means, const Eigen::MatrixXd& cov,
      unsigned int cases = 1) = 0;

  /**
   * Get the class name
   * @return Class name
   */
   virtual std::string name() const = 0;
  
 protected:
  int seed_ = static_cast<int>(
      std::time(nullptr)); /**< Seed value to use in random number generator */
};
}  // namespace numeric_utils

#endif  // _NUMERIC_UTILS_H_
