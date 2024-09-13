#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "nelder_mead.h"

std::vector<double> optimization::NelderMead::minimize(
    const std::vector<double>& initial_point, double delta,
    std::function<double(const std::vector<double>&)>& objective_function) {
  // Create vector of deltas with length equal to the number of dimensions
  std::vector<double> deltas(initial_point.size(), delta);
  // Call minmize with vector of deltas
  return minimize(initial_point, deltas, objective_function);
}

std::vector<double> optimization::NelderMead::minimize(
    const std::vector<double>& initial_point, const std::vector<double>& deltas,
    std::function<double(const std::vector<double>&)>& objective_function) {
  // Set the number of dimensions
  num_dimensions_ = initial_point.size();
  // Initialize matrix that expands initial simplex in different directions
  // using input deltas
  std::vector<std::vector<double>> simplex(
      num_dimensions_ + 1, std::vector<double>(num_dimensions_));
  for (unsigned int i = 0; i < num_dimensions_ + 1; ++i) {
    for (unsigned int j = 0; j < num_dimensions_; ++j) {
      simplex[i][j] = initial_point[j];
    }

    if (i != 0) {
      simplex[i][i - 1] = simplex[i][i - 1] + deltas[i - 1];
    }
  }

  // Call minimize with matrix definining initial simplex
  return minimize(simplex, objective_function);
}

std::vector<double> optimization::NelderMead::minimize(
    const std::vector<std::vector<double>>& initial_simplex,
    std::function<double(const std::vector<double>&)>& objective_function) {

  // Initialize variables
  unsigned int index_high, index_low, index_next_high;
  num_dimensions_ = initial_simplex[0].size();
  num_points_ = initial_simplex.size();

  func_vals_.resize(num_points_);
  simplex_ = initial_simplex;
  std::vector<double> evaluation_point(num_dimensions_);
  std::vector<double> simplex_mins(num_dimensions_);
  std::vector<double> centroids(num_dimensions_);

  // Evaluate objective function at all points in simplex
  for (unsigned int i = 0; i < num_points_; ++i) {
    for (unsigned int j = 0; j < num_dimensions_; ++j) {
      evaluation_point[j] = simplex_[i][j];
    }
    func_vals_[i] = objective_function(evaluation_point);
  }

  num_evals_ = 0;

  centroids = calc_centroid(simplex_, num_dimensions_, num_points_);
  
  // centroids = calc_centroid();

  // Iterate until specified tolerance is achieved or maximum number of
  // iterations is exceeded
  while (true) {
    index_low = 0;
    // Order points from best to worst
    index_high = func_vals_[0] > func_vals_[1] ? (index_next_high = 1, 0)
                                               : (index_next_high = 0, 1);

    for (unsigned int i = 0; i < num_points_; ++i) {
      // Check if function value is best than current low
      if (func_vals_[i] <= func_vals_[index_low]) {
        index_low = i;
      }
      // Check if function value is worse than current worst
      if (func_vals_[i] > func_vals_[index_high]) {
        index_next_high = index_high;
        index_high = i;
        // Check if function value is worse than next-worst, but not worst
      } else if (func_vals_[i] > func_vals_[index_next_high] &&
                 i != index_high) {
        index_next_high = i;
      }
    }

    // Calculate function tolerance
    double tolerance =
        2.0 * std::abs(func_vals_[index_high] - func_vals_[index_low]) /
        (std::abs(func_vals_[index_high]) + std::abs(func_vals_[index_low]) +
         EPSILON_);

    // Check if tolerance is sufficiently small
    if (tolerance < function_tol_) {
      std::swap(func_vals_[0], func_vals_[index_low]);
      simplex_[0].swap(simplex_[index_low]);
      simplex_mins = simplex_[0];
      func_min_ = func_vals_[0];

      return simplex_mins;
    }

    if (num_evals_ >= MAX_ITERS_) {
      std::swap(func_vals_[0], func_vals_[index_low]);
      simplex_[0].swap(simplex_[index_low]);
      simplex_mins = simplex_[0]; 
      func_min_ = func_vals_[0];
      // std::cerr << "\nWARNING: In optimization::NelderMead::minimize: Max "
      //              "iterations exceeded, returning current "
      //              "minimum location with function value of "
      //           << func_min_ << std::endl;

      return simplex_mins;      
    }

    num_evals_ += 2;

    // Start iteration by first extrapolating by a factor -1 through the face of
    // the simplex across from the high point--reflect simplex from high point
    double reflection = reflect(simplex_, func_vals_, centroids, index_high,
                                -1.0, objective_function);

    // Reflection gives better result than best point, so extrapolate again with
    // factor 2.0
    if (reflection <= func_vals_[index_low]) {
      reflection = reflect(simplex_, func_vals_, centroids, index_high, 2.0,
                           objective_function);
      // Reflection is worse than next-highest, so do 1-D contraction to find
      // intermediate lower point
    } else if (reflection >= func_vals_[index_next_high]) {
      double func_next_high = func_vals_[index_next_high];
      reflection = reflect(simplex_, func_vals_, centroids, index_high, 0.5,
                           objective_function);

      // Worst point is not going away, so contract around best point
      if (reflection >= func_next_high) {
        for (unsigned int i = 0; i < num_points_; ++i) {
          if (i != index_low) {
            for (unsigned int j = 0; j < num_dimensions_; ++j) {
              centroids[j] = 0.5 * (simplex_[i][j] + simplex_[index_low][j]);
              simplex_[i][j] = centroids[j];
            }
            func_vals_[i] = objective_function(centroids);
          }
        }
        num_evals_ += num_dimensions_;
        centroids = calc_centroid(simplex_, num_dimensions_, num_points_);
      }
    } else {
      --num_evals_;
    }
  }
}

double optimization::NelderMead::reflect(
    std::vector<std::vector<double>>& simplex,
    std::vector<double>& objective_vals, std::vector<double>& centroids,
    unsigned int index_worst, double factor,
    std::function<double(const std::vector<double>&)>& objective_function) {

  std::vector<double> evaluations(num_dimensions_);

  double factor1 = (1.0 - factor) / static_cast<double>(num_dimensions_);
  double factor2 = factor1 - factor;

  for (unsigned int j = 0; j < num_dimensions_; ++j) {
    evaluations[j] = centroids[j] * factor1 - simplex[index_worst][j] * factor2;
  }

  double objective_value = objective_function(evaluations);

  if (objective_value < objective_vals[index_worst]) {
    objective_vals[index_worst] = objective_value;

    for (unsigned int j = 0; j < num_dimensions_; ++j) {
      centroids[j] += evaluations[j] - simplex[index_worst][j];
      simplex[index_worst][j] = evaluations[j];
    }
  }

  return objective_value;
}

std::vector<double> optimization::NelderMead::calc_centroid(
    const std::vector<std::vector<double>>& simplex,
    unsigned int num_dimensions, unsigned int num_points) const {

  std::vector<double> centroids(num_dimensions);

  for (unsigned int j = 0; j < num_dimensions; ++j) {
    double sum = 0.0;
    for (unsigned int i = 0; i < num_points; ++i) {
      sum += simplex[i][j];
    }

    centroids[j] = sum;
  }

  return centroids;
}
