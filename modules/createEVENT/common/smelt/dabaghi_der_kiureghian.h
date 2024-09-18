#ifndef _DABAGHI_DER_KIUREGHIAN_H_
#define _DABAGHI_DER_KIUREGHIAN_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "distribution.h"
#include "json_object.h"
#include "numeric_utils.h"
#include "stochastic_model.h"

namespace stochastic {
/** @enum stochastic::FaultType
 *  @brief is a strongly typed enum class representing the type of faulting
 */  
enum class FaultType {
  StrikeSlip, /**< Strike-slip fault */
  ReverseAndRevObliq /**< Reverse or reverse-oblique fault */
};

/** @enum stochastic::SimulationType
 *  @brief is a strongly typed enum class representing pulse-like proportion
 *  of ground motion
 */  
enum class SimulationType {
  PulseAndNoPulse, /**< pulse-like and non-pulse-like motions in proportion predicted by Shahi and Baker(2014) */
  Pulse, /**< only pulse-like */
  NoPulse /**< only non-pulse-like */
};

/**
 * Stochastic model for simulating near-fault ground motions. Based on the following
 * references:
 *   1. Dabaghi and Der Kiureghian (2014 PEER report) "Stochastic Modeling and Simulation of Near-Fault Ground Motions for Performance-Based Earthquake Engineering"
 *   2. Dabaghi and Der Kiureghian (2017 EESD) "Stochastic model for simulation of NF GMs"
 *   3. Dabaghi and Der Kiureghian (2018 EESD) "Simulation of orthogonal horizontal components of near-fault ground motion for specified EQ source and site characteristics"
 */
class DabaghiDerKiureghian : public StochasticModel {
 public:
  /**
   * @constructor Default constructor
   */
  DabaghiDerKiureghian() = default;

  /**
   * @constructor Construct near-fault ground motion model based on input
   * parameters
   * @param[in] faulting Type of faulting
   * @param[in] simulation_type Pulse-type of simulation
   * @param[in] moment_magnitude Moment magnitude of earthquake
   * @param[in] depth_to_rupt Depth to the top of the rupture plane in
   *               kilometers
   * @param[in] rupture_distance Closest distance from the site to the fault
   *               rupture in kilometers
   * @param[in] vs30 Soil shear wave velocity averaged over top 30 meters in
   *               meters per second
   * @param[in] s_or_d Directivity parameter s or d (km)--input the larger of
   *               the two
   * @param[in] theta_or_phi Directivity ange parameter theta or phi
   *               (degrees)--input corresponding value to s or d
   * @param[in] num_sims Number of simulated ground motion time histories that
   *               should be generated (number of different model parameter realizations)
   * @param[in] num_realizations Number of realizations of non-stationary, modulated, filtered
   *               white noise per set of model parameters
   * @param[in] truncate Boolean indicating whether to truncate and baseline correct
   *               synthetic motion
   */
  DabaghiDerKiureghian(FaultType faulting, SimulationType simulation_type,
                       double moment_magnitude, double depth_to_rupt,
                       double rupture_distance, double vs30, double s_or_d,
                       double theta_or_phi, unsigned int num_sims,
                       unsigned int num_realizations, bool truncate);

  /**
   * @constructor Construct near-fault ground motion model based on input
   * parameters
   * @param[in] faulting Type of faulting
   * @param[in] simulation_type Pulse-type of simulation
   * @param[in] moment_magnitude Moment magnitude of earthquake
   * @param[in] depth_to_rupt Depth to the top of the rupture plane in
   *               kilometers
   * @param[in] rupture_distance Closest distance from the site to the fault
   *               rupture in kilometers
   * @param[in] vs30 Soil shear wave velocity averaged over top 30 meters in
   *               meters per second
   * @param[in] s_or_d Directivity parameter s or d (km)--input the larger of
   *               the two
   * @param[in] theta_or_phi Directivity ange parameter theta or phi
   *               (degrees)--input corresponding value to s or d
   * @param[in] num_sims Number of simulated ground motion time histories that
   *               should be generated (number of different model parameter realizations)
   * @param[in] num_realizations Number of realizations of non-stationary, modulated, filtered
   *               white noise per set of model parameters
   * @param[in] truncate Boolean indicating whether to truncate and baseline correct
   *               synthetic motion
   * @param[in] seed_value Value to seed random variables with to ensure
   *               repeatability
   */
  DabaghiDerKiureghian(FaultType faulting, SimulationType simulation_type,
                       double moment_magnitude, double depth_to_rupt,
                       double rupture_distance, double vs30, double s_or_d,
                       double theta_or_phi, unsigned int num_sims,
                       unsigned int num_realizations, bool truncate, int seed_value);

  /**
   * @destructor Virtual destructor
   */
  virtual ~DabaghiDerKiureghian() {};

  /**
   * Delete copy constructor
   */
  DabaghiDerKiureghian(const DabaghiDerKiureghian&) = delete;

  /**
   * Delete assignment operator
   */
  DabaghiDerKiureghian& operator=(const DabaghiDerKiureghian&) = delete;

  /**
   * Generate ground motion time histories based on input parameters
   * and store outputs as JSON object. Throws exception if errors
   * are encountered during time history generation.
   * @param[in] event_name Name to assign to event
   * @param[in] units Indicates that time histories should be returned in
   *                  units of g. Defaults to false where time histories
   *                  are returned in units of m/s^2
   * @return JsonObject containing time histories
   */
  utilities::JsonObject generate(const std::string& event_name,
                                 bool units = false) override;

  /**
   * Generate ground motion time histories based on input parameters
   * and write results to file in JSON format. Throws exception if
   * errors are encountered during time history generation.
   * @param[in] event_name Name to assign to event
   * @param[in, out] output_location Location to write outputs to
   * @param[in] units Indicates that time histories should be returned in
   *                  units of g. Defaults to false where time histories
   *                  are returned in units of m/s^2
   * @return Returns true if successful, false otherwise
   */
  bool generate(const std::string& event_name,
                const std::string& output_location,
                bool units = false) override;

  /**
   * Generates proportion of motions that should be pulse-like based on total
   * number of simulations and probability of those motions containing a pulse
   * following pulse probability model developed by Shahi & Baker (2014)
   * @param[in] num_sims Total number of simulations that should be generated
   * @return Total number of pulse-like motions
   */
  unsigned int simulate_pulse_type(unsigned num_sims) const;

  /**
   * Simulate model parameters for ground motions based on either pulse-like
   * or non-pulse-like behavior
   * @param[in] pulse_like Boolean indicating whether ground motions are
   *                       pulse-like
   * @param[in] num_sims Number of simulations to simulate model parameters for
   * @return Model parameters for ground motions
   */
  Eigen::MatrixXd simulate_model_parameters(bool pulse_like,
                                            unsigned int num_sims);

  /**
   * Compute the conditional mean values of the transformed model parameters
   * using regressiong coefficients and Equation 12 from Dabaghi & Der
   * Kiureghian (2017)
   * @param[in] pulse_like Boolean indicating whether ground motions are
   *                       pulse-like
   * @return Vector containing predicted model parameters
   */
  Eigen::VectorXd compute_transformed_model_parameters(bool pulse_like) const;

  /**
   * Transforms model parameters from normal space back to real space
   * @param[in] pulse_like Boolean indicating whether ground motions are
   *                       pulse-like
   * @param[in, out] parameters Vector of parameters in normal space. Transformed variables will be
   *                            stored in this vector.
   */
  void transform_parameters_from_normal_space(bool pulse_like, Eigen::VectorXd& parameters);

  /**
   * Calculate the inverse of double-exponential distribution
   * @param[in] probability Probability at which to evaluate inverse CDF
   * @param[in] param_a Distribution parameter
   * @param[in] param_b Distribution parameter
   * @param[in] param_c Distribution parameter
   * @param[in] lower_bound Lower bound for location
   */
  double inv_double_exp(double probability, double param_a, double param_b,
                        double param_c, double lower_bound) const;

  /**
   * Simulate near-fault ground motion given model parameters and whether motion
   * is pulse-like or not.
   * @param[in] pulse_like Boolean indicating whether ground motions are
   *                       pulse-like
   * @param[in] parameters Vector of model parameters to use for ground motion
   *                       simulation
   * @param[in,out] accel_comp_1 Simulated near-fault ground motion components
   *                             in direction 1. Outputs are written here.
   * @param[in,out] accel_comp_2 Simulated near-fault ground motion components
   *                             in direction 2. Outputs are written here.
   * @param[in] num_gms Number of ground motions that should be generated.
   *                    Defaults to 1.
   */
  void simulate_near_fault_ground_motion(
      bool pulse_like, const Eigen::VectorXd& parameters,
      std::vector<std::vector<double>>& accel_comp_1,
      std::vector<std::vector<double>>& accel_comp_2,
      unsigned int num_gms = 1) const;

  /**
   * Backcalculate modulating parameters given Arias Intesity and duration parameters
   * @param[in] q_params Vector containing Ia, D595, D05, and D030
   * @param[in] t0 Initial time. Defaults to 0.0.
   * @return Vector containing parameters alpha, beta, tmaxq, and c
   */
  Eigen::VectorXd backcalculate_modulating_params(
      const Eigen::VectorXd& q_params, double t0 = 0.0) const;

  /**
   * Simulate modulated filtered white noise process
   * @param[in] modulating_params Modulating parameters
   * @param[in] filter_params Filtering parameters
   * @param[in] num_steps Total number of time steps to be taken
   * @param[in] num_gms Number of ground motions that should be generated.
   *                    Defaults to 1.
   * @return Vector of vectors containing time history of simulated modulate
   *         filtered white noise
   */
  Eigen::MatrixXd simulate_white_noise(const Eigen::VectorXd& modulating_params,
                                       const Eigen::VectorXd& filter_params,
                                       unsigned int num_steps,
                                       unsigned int num_gms = 1) const;

  /**
   * This function defines an error measure based on matching times of the 5%,
   * 30%, and 95% Arias intensity of the target ground motion and corresponding
   * modulating function q(t) with parameters alpha_q_sub as defined in Eq. 4
   * of Reference 2. This is used to back-calculate the modulating function
   * parameters by minimizing the corresponding error measure. Input to returned
   * function is a vector containing alpha, beta, and t_max_q.
   * @param[in] parameters Modulating function parameters: alpha, beta, and t_max_q
   * @param[in] d05_target Time from t0 to time of 5% Arias intensity of target
   *                       motion
   * @param[in] d030_target Time from t0 to time of 30% Arias intensity of
   *                        target motion
   * @param[in] d095_target Time from t0 to time of 95% Arias intensity of
   *                        target motion
   * @param[in] t0 Start time of modulating function and of target ground motion
   * @return ERrro in modulating function
   */
  double calc_parameter_error(const std::vector<double>& parameters,
                              double d05_target, double d030_target,
                              double d095_target, double t0) const;

  /**
   * Calculate values of modulating function given function parameters
   * @param[in] num_steps Total number of time steps to be taken
   * @param[in] t0 Initial time
   * @param[in] parameters Modulating function parameters
   * @return Vector containing time series of modulating function values
   */
  std::vector<double> calc_modulating_func(
      unsigned int num_steps, double t0,
      const Eigen::VectorXd& parameters) const;

  /**
   * Calculate the time at which the input percentage of the Arias intensity
   * is reached
   * @param[in] acceleration Acceleration time history
   * @param[in] percentage Percentage of Arias intensity to be reached
   * @return Time at which input percentage of Arias intensity is reached
   */
  double calc_time_to_intensity(const std::vector<double>& acceleration,
                                double percentage) const;

  /**
   * Calculate the linearly varying filter function (in rad/sec) given the
   * filter function parameters (in Hz) and the times of 1%, 30%(mid) and 99%
   * Arias Intensity (AI)
   * @param[in] num_steps Number of time steps in time history
   * @param[in] filter_params Filter function parameters [fmid, f_slope] (in
   *                          Hz), tmid is defined as the time of 30% AI
   * @param[in] t01 Time of 1% of AI of the modulating function (and in an
   *                average sense of the simulated GM)
   * @param[in] tmid Time of 30% of AI of the modulating function (and in an
   *                average sense of the simulated GM)
   * @param[in] t99 Time of 99% of AI of the modulating function (and in an
   *                average sense of the simulated GM)
   */
  std::vector<double> calc_linear_filter(unsigned int num_steps,
                                         const Eigen::VectorXd& filter_params,
                                         double t01, double tmid,
                                         double t99) const;

  /**
   * Calculate impulse response filter based on time series, input filter,
   * and filter parameter zeta
   * @param[in] num_steps Number of time steps in time history
   * @param[in] input_filter Input filter coefficients to use in impulse
   *                         response
   * @param[in] zeta Filter parameter
   * @return Impulse response filter
   */
  Eigen::MatrixXd calc_impulse_response_filter(
      unsigned int num_steps, const std::vector<double>& input_filter,
      double zeta) const;

  /**
   * Filters input acceleration time history in frequency domain using
   * acausal high-pass Butterworth filter
   * @param[in] accel_history Acceleration time history to filter
   * @param[in] freq_corner Corner frequency
   * @param[in] filter_order Order of filter
   * @return Filtered time history
   */
  std::vector<double> filter_acceleration(const Eigen::VectorXd& accel_history,
                                          double freq_corner,
                                          unsigned int filter_order) const;

  /**
   * Calculate the pulse acceleration based on the modified Mavroeidis and
   * Papageorgiou model
   * @param[in] num_steps Number of time steps in time history
   * @param[in] parameters Vector of model parameters to use for ground motion
   *                       simulation
   * @return Time history of pulse acceleration
   */
  std::vector<double> calc_pulse_acceleration(
      unsigned int num_steps, const Eigen::VectorXd& parameters) const;

  /**
   * Truncate acceleration time histories at the beginning and/or end where
   * displacement amplitudes are almost zero effectively zero
   * @param[in, out] accel_comp_1 Component 1 of acceleration time history to
   *                              truncate
   * @param[in, out] accel_comp_2 Component 2 of acceleration time history to
   *                              truncate
   * @param[in] gfactor Factor to convert acceleration to cm/s^2
   * @param[in] amplitude_lim Displacement amplitude limit in cm below which to
   *                          apply truncation. Defaults to 0.2cm
   * @param[in] pgd_lim Ratio of peak ground displacement below which to
   *                    truncate. Defaults to 0.01.
   *
   */
  void truncate_time_histories(std::vector<std::vector<double>>& accel_comp_1,
                               std::vector<std::vector<double>>& accel_comp_2,
                               double gfactor, double amplitude_lim = 0.2,
                               double pgd_lim = 0.01) const;

  /**
   * Baseline correct acceleration time histories by fitting a polynomial
   * starting from the 2nd degree of the displacement time series
   * @param[in, out] time_history Acceleration time history to truncate
   * @param[in] gfactor Factor to convert acceleration to cm/s^2
   * @param[in] order Order of the polynomial fitted to the displacement time
   *                  series
   */
  void baseline_correct_time_history(std::vector<double>& time_history,
                                     double gfactor, unsigned int order) const;

  /**
   * Convert input time history to units of g or m/s^2
   * @param[in, out] time_history Time history to convert units for
   * @param[in] units If true, converts to units of g, otherwise to m/s^2
   */
  void convert_time_history_units(std::vector<double>& time_history,
                                  bool units) const;  

 private:
  FaultType faulting_;      /**< Enum for type of faulting for scenario */
  SimulationType sim_type_; /**< Enum for pulse-like nature of ground motion */
  double moment_magnitude_; /**< Moment magnitude for scenario */
  double depth_to_rupt_; /**< Depth to the top of the rupture plane (km) */
  double rupture_dist_; /**< Closest-to-site rupture distance in kilometers */
  double vs30_; /**< Soil shear wave velocity averaged over top 30 meters in
                   meters per second */
  double s_or_d_; /**< Directivity parameter s or d (km) */
  double theta_or_phi_; /**< Directivity angle parameter theta or phi */
  bool truncate_; /**< Indicates whether to truncate and baseline correct motion */
  unsigned int num_sims_pulse_; /**< Number of pulse-like simulated ground
                             motion time histories that should be generated */
  unsigned int num_sims_nopulse_; /**< Number of no-pulse-like simulated ground
                             motion time histories that should be generated */
  unsigned int num_realizations_; /**< Number of realizations of model parameters */
  int seed_value_; /**< Integer to seed random distributions with */
  double time_step_; /**< Temporal discretization. Set to 0.005 seconds */
  double start_time_ = 0.0; /**< Start time of ground motion */
  Eigen::VectorXd std_dev_pulse_; /**< Pulse-like parameter standard deviation */
  Eigen::VectorXd std_dev_nopulse_; /**< No-pulse-like parameter standard deviation */
  Eigen::MatrixXd corr_matrix_pulse_; /**< Pulse-like parameter correlation matrix */
  Eigen::MatrixXd corr_matrix_nopulse_; /**< No-pulse-like parameter correlation matrix */
  Eigen::MatrixXd beta_distribution_pulse_; /**< Beta distrubution parameters for pulse-like motion */
  Eigen::MatrixXd beta_distribution_nopulse_; /**< Beta distrubution parameters for no-pulse-like motion */
  Eigen::VectorXd params_lower_bound_;        /**< Lower bound for marginal distributions fitted to params
						 (Table 5 in Dabaghi & Der Kiureghian, 2017) */
  Eigen::VectorXd params_upper_bound_;        /**< Upper bound for marginal distributions fitted to params
						 (Table 5 in Dabaghi & Der Kiureghian, 2017) */
  Eigen::VectorXd params_fitted1_; /** Fitted distribution parameters from Table 5 (Dabaghi & Der Kiureghian, 2017) */
  Eigen::VectorXd params_fitted2_; /** Fitted distribution parameters from Table 5 (Dabaghi & Der Kiureghian, 2017) */
  Eigen::VectorXd params_fitted3_; /** Fitted distribution parameters from Table 5 (Dabaghi & Der Kiureghian, 2017) */  
  const double magnitude_baseline_ = 6.5; /**< Baseline regression factor for magnitude */ 
  const double c6_ = 6.0 ; /**< This factor is set to avoid non-linearity in regression */
  std::shared_ptr<numeric_utils::RandomGenerator>
      sample_generator_; /**< Multivariate normal random number generator */
};
}  // namespace stochastic

#endif  // _DABAGHI_DER_KIUREGHIAN_H_
