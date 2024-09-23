#include <cmath>
#include <complex>
#include <ctime>
#include <string>
// Boost random generator
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
// Eigen dense matrices
#include <Eigen/Dense>

#include "function_dispatcher.h"
#include "json_object.h"
#include "numeric_utils.h"
#include "wittig_sinha.h"

stochastic::WittigSinha::WittigSinha(std::string exposure_category,
                                     double gust_speed, double height,
                                     unsigned int num_floors, double total_time)
    : StochasticModel(),
      exposure_category_{exposure_category},
      gust_speed_{gust_speed * 0.44704}, // Convert from mph to m/s
      bldg_height_{height},
      num_floors_{num_floors},
      seed_value_{std::numeric_limits<int>::infinity()},
      local_x_{std::vector<double>(1, 1.0)},
      local_y_{std::vector<double>(1, 1.0)},
      freq_cutoff_{5.0},
      time_step_{1.0 / (2.0 * freq_cutoff_)} {
  model_name_ = "WittigSinha";
  num_times_ =
      static_cast<unsigned int>(std::ceil(total_time / time_step_)) % 2 == 0
          ? static_cast<unsigned int>(std::ceil(total_time / time_step_))
          : static_cast<unsigned int>(std::ceil(total_time / time_step_) + 1);

  // Calculate range of frequencies based on cutoff frequency
  num_freqs_ = num_times_ / 2;
  frequencies_.resize(num_freqs_);

  for (unsigned int i = 0; i < frequencies_.size(); ++i) {
    frequencies_[i] = (i + 1) * freq_cutoff_ / num_freqs_;
  }
  
  // Calculate heights of each floor
  heights_ = std::vector<double>(num_floors_);  
  heights_[0] = bldg_height_ / num_floors_;

  for (unsigned int i = 1; i < heights_.size(); ++i) {
    heights_[i] = heights_[i - 1] + bldg_height_ / num_floors_;
  }

  // Calculate velocity profile
  friction_velocity_ =
      Dispatcher<double, const std::string&, const std::vector<double>&, double,
                 double, std::vector<double>&>::instance()
          ->dispatch("ExposureCategoryVel", exposure_category, heights_, 0.4,
                     gust_speed, wind_velocities_);
}

stochastic::WittigSinha::WittigSinha(std::string exposure_category,
                                     double gust_speed, double height,
                                     unsigned int num_floors, double total_time,
                                     int seed_value)
    : WittigSinha(exposure_category, gust_speed, height, num_floors,
                  total_time)
{
  seed_value_ = seed_value;
}

stochastic::WittigSinha::WittigSinha(std::string exposure_category,
                                     double gust_speed,
                                     const std::vector<double>& heights,
                                     const std::vector<double>& x_locations,
                                     const std::vector<double>& y_locations,
                                     double total_time)
    : StochasticModel(),
      exposure_category_{exposure_category},
      gust_speed_{gust_speed * 0.44704}, // Convert from mph to m/s
      seed_value_{std::numeric_limits<int>::infinity()},
      heights_{heights},
      local_x_{x_locations},
      local_y_{y_locations},
      freq_cutoff_{5.0},
      time_step_{1.0 / (2.0 * freq_cutoff_)}
{
  model_name_ = "WittigSinha";
  num_times_ =
      static_cast<unsigned int>(std::ceil(total_time / time_step_)) % 2 == 0
          ? static_cast<unsigned int>(std::ceil(total_time / time_step_))
          : static_cast<unsigned int>(std::ceil(total_time / time_step_) + 1);

  // Calculate range of frequencies based on cutoff frequency
  num_freqs_ = num_times_ / 2;
  frequencies_.resize(num_freqs_);

  for (unsigned int i = 0; i < frequencies_.size(); ++i) {
    frequencies_[i] = i * freq_cutoff_ / num_freqs_;
  }

  // Calculate velocity profile
  friction_velocity_ =
      Dispatcher<double, const std::string&, const std::vector<double>&, double,
                 double, std::vector<double>&>::instance()
          ->dispatch("ExposureCategoryVel", exposure_category, heights_, 0.4,
                     gust_speed, wind_velocities_);  
}

stochastic::WittigSinha::WittigSinha(std::string exposure_category,
                                     double gust_speed,
                                     const std::vector<double>& heights,
                                     const std::vector<double>& x_locations,
                                     const std::vector<double>& y_locations,
                                     double total_time, int seed_value)
  : WittigSinha(exposure_category, gust_speed, heights, x_locations, y_locations, total_time)
{
  seed_value_ = seed_value;
}

utilities::JsonObject stochastic::WittigSinha::generate(const std::string& event_name, bool units) {
  // Initialize wind velocity vectors
  std::vector<std::vector<std::vector<std::vector<double>>>> wind_vels(
      local_x_.size(),
      std::vector<std::vector<std::vector<double>>>(
          local_y_.size(),
          std::vector<std::vector<double>>(
              heights_.size(), std::vector<double>(num_times_, 0.0))));

  Eigen::MatrixXcd complex_random_vals(num_freqs_, heights_.size());
  
  // Loop over heights to find time histories
  try {
    for (unsigned int i = 0; i < local_x_.size(); ++i) {
      for (unsigned int j = 0; j < local_y_.size(); ++j) {
        // Generate complex random numbers to use for calculation of discrete
        // time series
        complex_random_vals = complex_random_numbers();
        for (unsigned int k = 0; k < heights_.size(); ++k) {
          wind_vels[i][j][k] = gen_location_hist(complex_random_vals, k, units);
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "\nERROR: In stochastic::WittigSinha::generate: "
              << e.what() << std::endl;
  }

  // Create JsonObject for event
  auto event = utilities::JsonObject();
  event.add_value("dT", time_step_);
  event.add_value("numSteps", num_times_);
  
  // Consider case when only looking at floor loads, so only have time histories as
  // one location along the z-axis
  if (local_x_.size() == 1 && local_y_.size() == 1) {
    // Arrays of patterns and time histories for each floor
    std::vector<utilities::JsonObject> pattern_array(heights_.size());
    std::vector<utilities::JsonObject> event_array(1);
    std::vector<utilities::JsonObject> time_history_array(heights_.size());
    auto time_history = utilities::JsonObject();
    event_array[0].add_value("type", "Wind");
    event_array[0].add_value("subtype", model_name_);

    for (unsigned int i = 0; i < heights_.size(); ++i) {
      // Create pattern
      pattern_array[i].add_value("name", std::to_string(i + 1));
      pattern_array[i].add_value("timeSeries", std::to_string(i + 1));
      pattern_array[i].add_value("type", "WindFloorLoad");
      pattern_array[i].add_value("floor", std::to_string(i + 1));
      pattern_array[i].add_value("dof", 1);
      pattern_array[i].add_value("profileVelocity", wind_velocities_[i]);
      
      // Create time histories
      time_history.add_value("name", std::to_string(i + 1));
      time_history.add_value("dT", time_step_);
      time_history.add_value("type", "Value");
      time_history.add_value("data", wind_vels[0][0][i]);
      time_history_array[i] = time_history;
      time_history.clear();
    }
    
    event_array[0].add_value("timeSeries", time_history_array);
    event_array[0].add_value("pattern", pattern_array);
    event.add_value("Events", event_array);   
  } else {
    throw std::runtime_error(
        "ERROR: In stochastic::WittigSinha::generate: Currently, only supports "
        "time histories along z-axis at single location\n");
  }

  return event;
}

bool stochastic::WittigSinha::generate(const std::string& event_name,
                                       const std::string& output_location,
                                       bool units) {

  bool status = true;
  // Generate time histories at specified locations
  try {
    auto json_output = generate(event_name, units);
    json_output.write_to_file(output_location);
  } catch (const std::exception& e) {
    std::cerr << e.what();
    status = false;
    throw;
  }

  return status;
}

Eigen::MatrixXd stochastic::WittigSinha::cross_spectral_density(double frequency) const {
  // Coefficient for coherence function
  double coherence_coeff = 10.0;
  Eigen::MatrixXd cross_spectral_density =
      Eigen::MatrixXd::Zero(heights_.size(), heights_.size());
  
  for (unsigned int i = 0; i < cross_spectral_density.rows(); ++i) {
    cross_spectral_density(i, i) =
        200.0 * friction_velocity_ * friction_velocity_ * heights_[i] /
        (wind_velocities_[i] *
         std::pow(1.0 + 50.0 * frequency * heights_[i] / wind_velocities_[i],
                  5.0 / 3.0));
  }

  for (unsigned int i = 0; i < cross_spectral_density.rows(); ++i) {
    for (unsigned int j = i + 1; j < cross_spectral_density.cols(); ++j) {
      cross_spectral_density(i, j) =
          std::sqrt(cross_spectral_density(i, i) *
                    cross_spectral_density(j, j)) *
          std::exp(-coherence_coeff * frequency *
                   std::abs(heights_[i] - heights_[j]) /
                   (0.5 * (wind_velocities_[i] + wind_velocities_[j]))) *
          0.999;
    }
  }

  // Get diagonal of cross spectral density matrix--avoids compiler errors where type
  // of diagonal matrix is not correctly deduced
  Eigen::MatrixXd diag_mat = cross_spectral_density.diagonal().asDiagonal();

  return cross_spectral_density.transpose() + cross_spectral_density - diag_mat;
}

Eigen::MatrixXcd stochastic::WittigSinha::complex_random_numbers() const {
  // Construct random number generator for standard normal distribution
  static unsigned int history_seed = static_cast<unsigned int>(std::time(nullptr));
  history_seed = history_seed + 10;

  auto generator =
    seed_value_ != std::numeric_limits<int>::infinity()
    ? boost::random::mt19937(static_cast<unsigned int>(seed_value_ + 10))
    : boost::random::mt19937(history_seed);
  
  boost::random::normal_distribution<> distribution;
  boost::random::variate_generator<boost::random::mt19937&,
                                   boost::random::normal_distribution<>>
      distribution_gen(generator, distribution);

  // Generate white noise consisting of complex numbers
  Eigen::MatrixXcd white_noise(heights_.size(), num_freqs_);

  for (unsigned int i = 0; i < white_noise.rows(); ++i) {
    for (unsigned int j = 0; j < white_noise.cols(); ++j) {
      white_noise(i, j) = std::complex<double>(
          distribution_gen() * std::sqrt(0.5),
          distribution_gen() * std::sqrt(std::complex<double>(-0.5)).imag());
    }
  }

  // Iterator over all frequencies and generate complex random numbers
  // for discrete time series simulation
  Eigen::MatrixXd cross_spec_density_matrix(heights_.size(), heights_.size());
  Eigen::MatrixXcd complex_random(num_freqs_, heights_.size());

  for (unsigned int i = 0; i < frequencies_.size(); ++i) {
    // Calculate cross-spectral density matrix for current frequency
    cross_spec_density_matrix = cross_spectral_density(frequencies_[i]);

    // Find lower Cholesky factorization of cross-spectral density
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lower_cholesky;

    try {
      auto llt = cross_spec_density_matrix.llt();
      lower_cholesky = llt.matrixL();

      if (llt.info() == Eigen::NumericalIssue) {
        throw std::runtime_error(
            "\nERROR: In stochastic::WittigSinha::generate method: Cross-Spectral Density "
            "matrix is not positive semi-definite\n");
      }
    } catch (const std::exception& e) {
      std::cerr << "\nERROR: In time history generation: " << e.what()
                << std::endl;
    }
    
    // This is Equation 5(a) from Wittig & Sinha (1975)
    complex_random.row(i) = num_freqs_ *
                            std::sqrt(2.0 * freq_cutoff_ / num_freqs_) *
                            lower_cholesky * white_noise.col(i);
  }

  return complex_random;
}

std::vector<double> stochastic::WittigSinha::gen_location_hist(
    const Eigen::MatrixXcd& random_numbers, unsigned int column_index,
    bool units) const {

  // This following block implements what is expressed in Equations 7 & 8
  Eigen::VectorXcd complex_full_range = Eigen::VectorXcd::Zero(2 * num_freqs_);

  complex_full_range.segment(1, num_freqs_) =
      random_numbers.block(0, column_index, num_freqs_, 1);

  complex_full_range.segment(num_freqs_ + 1, num_freqs_ - 1) =
      random_numbers.block(0, column_index, num_freqs_ - 1, 1)
          .reverse()
          .conjugate();
 
  complex_full_range(num_freqs_) = std::abs(random_numbers(num_freqs_ - 1, column_index)); 

  // Calculate wind speed using real portion of inverse Fast Fourier Transform
  // full range of random numbers
  std::vector<double> node_time_history(complex_full_range.size());
  numeric_utils::inverse_fft(complex_full_range, node_time_history);

  // Check if time histories need to be converted to ft/s
  if (units) {
    for (auto & val : node_time_history) {
      val = val * 3.28084;
    }
  }
  
  return node_time_history;
}
