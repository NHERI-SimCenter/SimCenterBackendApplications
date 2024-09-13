#include <Eigen/Dense>
#include "beta_dist.h"
#include "configure.h"
#include "dabaghi_der_kiureghian.h"
#include "factory.h"
#include "filter.h"
#include "function_dispatcher.h"
#include "inv_gauss_dist.h"
#include "lognormal_dist.h"
#include "numeric_utils.h"
#include "normal_dist.h"
#include "normal_multivar.h"
#include "students_t_dist.h"
#include "uniform_dist.h"
#include "vlachos_et_al.h"
#include "wind_profile.h"
#include "window.h"
#include "wittig_sinha.h"

void config::initialize() {
  // RANDOM VARIABLE GENERATION
  // Register multivariate normal distribution random number generator
  static Register<numeric_utils::RandomGenerator, numeric_utils::NormalMultiVar>
      normal_multivar_default("MultivariateNormal");
  static Register<numeric_utils::RandomGenerator, numeric_utils::NormalMultiVar,
                  int>
      normal_multivar("MultivariateNormal");

  // DISTRIBUTION TYPES
  // Register normal distribution
  static Register<stochastic::Distribution, stochastic::NormalDistribution,
                  double, double>
      normal_dist("NormalDist");
  // Register lognormal distribution
  static Register<stochastic::Distribution, stochastic::LognormalDistribution,
                  double, double>
      lognormal_dist("LognormalDist");
  // Register inverse Gaussian distribution
  static Register<stochastic::Distribution,
                  stochastic::InverseGaussianDistribution, double, double>
      inv_gauss_dist("InverseGaussianDist");
  // Register beta distribution
  static Register<stochastic::Distribution, stochastic::BetaDistribution,
                  double, double>
      beta_dist("BetaDist");
  // Register Student's t distribution
  static Register<stochastic::Distribution, stochastic::StudentstDistribution,
                  double, double, double>
      student_t_dist("StudentstDist");
  // Register uniform distribution
  static Register<stochastic::Distribution, stochastic::UniformDistribution,
                  double, double>
      uniform_dist("UniformDist");

  // STOCHASTIC MODELS
  // Earthquake
  static Register<stochastic::StochasticModel, stochastic::VlachosEtAl, double,
                  double, double, double, unsigned int, unsigned int>
      vlachos_et_al("VlachosSiteSpecificEQ");
  static Register<stochastic::StochasticModel, stochastic::VlachosEtAl, double,
                  double, double, double, unsigned int, unsigned int, int>
      vlachos_et_al_seed("VlachosSiteSpecificEQ");
  static Register<stochastic::StochasticModel, stochastic::DabaghiDerKiureghian,
                  stochastic::FaultType, stochastic::SimulationType, double,
                  double, double, double, double, double, unsigned int,
                  unsigned int, bool>
      dabaghi_der_kiureghian("DabaghiDerKiureghianNFGM");
  static Register<stochastic::StochasticModel, stochastic::DabaghiDerKiureghian,
                  stochastic::FaultType, stochastic::SimulationType, double,
                  double, double, double, double, double, unsigned int,
                  unsigned int, bool, int>
      dabaghi_der_kiureghian_seed("DabaghiDerKiureghianNFGM");

  // Wind
  static Register<stochastic::StochasticModel, stochastic::WittigSinha,
                  std::string, double, double, unsigned int, double>
      wittig_sinha_equal_floors("WittigSinhaDiscreteFreqWind");
  static Register<stochastic::StochasticModel, stochastic::WittigSinha,
                  std::string, double, double, unsigned int, double, int>
      wittig_sinha_equal_floors_seed("WittigSinhaDiscreteFreqWind");
  static Register<stochastic::StochasticModel, stochastic::WittigSinha,
                  std::string, double, const std::vector<double>&,
                  const std::vector<double>&, const std::vector<double>&,
                  double>
      wittig_sinha_unequal_floors("WittigSinhaDiscreteFreqWind");
  static Register<stochastic::StochasticModel, stochastic::WittigSinha,
                  std::string, double, const std::vector<double>&,
                  const std::vector<double>&, const std::vector<double>&,
                  double, int>
      wittig_sinha_unequal_floors_seed("WittigSinhaDiscreteFreqWind");  

  // WINDOW FUNCTIONS
  // Register Hann window
  static DispatchRegister<Eigen::VectorXd, unsigned int> hann_window_function(
      "HannWindow", signal_processing::hann_window);

  // FILTER FUNCTIONS
  // Register highpass Butterworth filter
  static DispatchRegister<std::vector<std::vector<double>>, int, double>
      hp_butterworth_function("HighPassButter",
                              signal_processing::hp_butterworth());

  // Register filter impulse response
  static DispatchRegister<std::vector<double>, std::vector<double>,
                          std::vector<double>, int, int>
      filter_impulse_response("ImpulseResponse",
                              signal_processing::impulse_response());

  // Register acausal highpass Butterwork filter
  static DispatchRegister<std::vector<double>, double, double, unsigned int,
                          unsigned int>
      acausal_highpass_filter("AcausalHighpassButterworth",
                              signal_processing::acausal_highpass_filter());

  // WIND VELOCITY PROFILES
  // Exposure category-based velocity profile using power law
  static DispatchRegister<double, const std::string&,
                          const std::vector<double>&, double, double,
                          std::vector<double>&>
      exposure_category_vel("ExposureCategoryVel",
                            wind::exposure_category_velocity());
}
