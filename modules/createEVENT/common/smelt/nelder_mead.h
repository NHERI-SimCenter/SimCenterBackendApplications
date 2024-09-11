#ifndef _NELDER_MEAD_H_
#define _NELDER_MEAD_H_

#include <functional>
#include <limits>
#include <vector>

/**
 * Optimization utilities
 */
namespace optimization {

/**
 * Class that implements Nelder-Mead algorithm for multidimensional
 * unconstrained optimization. Based on implementation presented in
 * Press et al. (2007) - "Numerical Recipes"
 */
class NelderMead {
 public:
  /**
   * @constructor Default constructor
   */
  NelderMead() = default;

  /**
   * @constructor Construct with input function tolerance
   * @param[in] function_tolerance Tolerance in consecutive function evaluations
   *                               for convergence
   */
  NelderMead(double function_tolerance)
      : function_tol_{function_tolerance},
        func_min_{std::numeric_limits<double>::infinity()} {};

  /**
   * @destructor Virtual destructor
   */
  virtual ~NelderMead(){};

  /**
   * Delete copy constructor
   */
  NelderMead(const NelderMead&) = delete;

  /**
   * Delete assignment operator
   */
  NelderMead& operator=(const NelderMead&) = delete;

  /**
   * Minimize the input objective function given initial point and step size
   * @param[in] initial_point Initial values to use for each dimension
   * @param[in] delta Single step size to use for each dimension
   * @param[in] objective_function Function to minimize
   * @return Location of minimum
   */
  std::vector<double> minimize(
      const std::vector<double>& initial_point, double delta,
      std::function<double(const std::vector<double>&)>& objective_function);

  /**
   * Minimize the input objective function given initial point and step sizes
   * @param[in] initial_point Initial values to use for each dimension
   * @param[in] deltas Vector of step sizes to use for dimensions
   * @param[in] objective_function Function to minimize
   * @return Location of minimum
   */
  std::vector<double> minimize(
      const std::vector<double>& initial_point, const std::vector<double>& deltas,
      std::function<double(const std::vector<double>&)>& objective_function);

  /**
   * Get the minimum value of the objective function
   * @return Minimum value of objective function
   */
  double get_minimum() const { return func_min_; };

  /**
   * Minimize the input objective function given initial simplex
   * @tparam Tfunc_returntype Return type of objective function
   * @tparam Tfunc_args Objective function arguments
   * @param[in] initial_simplex Initial simplex to start optimization
   * @param[in] objective_function Function to minimize
   * @return Location of minimum
   */
  std::vector<double> minimize(
      const std::vector<std::vector<double>>& initial_simplex,
      std::function<double(const std::vector<double>&)>& objective_function);

 private:
  /**
   * Calculates the centroid of all points and returns them as a vector
   * @param[in] simplex Matrix describing simplex
   * @return Vector containing centroids
   */
  std::vector<double> calc_centroid(
      const std::vector<std::vector<double>>& simplex,
      unsigned int num_dimensions, unsigned int num_points) const;

  /**
   * Exptrapolate by input factor through the face of the simplex across from
   * the high point. Replaces high point if the new point is better.
   * @param[in] simplex Matrix describing simplex
   * @param[in] objective_vals Vector of objective values based on simplex
   * @param[in] centroids Vector containing centroids
   * @param[in] index_worst Index of worst value
   * @param[in] factor Factor by which to extrapolate
   * @param[in] objective_function Objective function to minimize
   * @return Objective value
   */
  double reflect(
      std::vector<std::vector<double>>& simplex,
      std::vector<double>& objective_vals, std::vector<double>& centroids,
      unsigned int index_worst, double factor,
      std::function<double(const std::vector<double>&)>& objective_function);

  double function_tol_;           /**< Function tolerance for convergence */
  unsigned int num_evals_;        /**< Number of function evaluations */
  unsigned int num_points_;       /**< Number of points */
  unsigned int num_dimensions_;   /**< Number of dimensions */
  double func_min_;               /**< Objective function minimum */
  std::vector<double> func_vals_; /**< Function values at vertices */
  std::vector<std::vector<double>> simplex_; /**< Current simplex */
  const double EPSILON_ = 1.0e-10;           /**< Tolerance */
  const unsigned int MAX_ITERS_ = 10000; /**< Maximum number of iterations */
};
}  // namespace optimization

#endif  // _NELDER_MEAD_H_
