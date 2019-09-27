#include <memory>
#include <string>
#include <dabaghi_der_kiureghian.h>
#include <json_object.h>
#include <factory.h>
#include <stochastic_model.h>
#include "eq_generator.h"

EQGenerator::EQGenerator(std::string model_name, double moment_magnitude,
                         double rupture_dist, double vs30) {
  eq_model_ = Factory<stochastic::StochasticModel, double, double, double,
                      double, unsigned int, unsigned int>::instance()
                  ->create(model_name, std::move(moment_magnitude),
                           std::move(rupture_dist), std::move(vs30),
                           std::move(0.0), std::move(1), std::move(1));
}

EQGenerator::EQGenerator(std::string model_name, double moment_magnitude,
                         double rupture_dist, double vs30, int seed) {
  eq_model_ =
      Factory<stochastic::StochasticModel, double, double, double, double,
              unsigned int, unsigned int, int>::instance()
          ->create(model_name, std::move(moment_magnitude),
                   std::move(rupture_dist), std::move(vs30), std::move(0.0),
                   std::move(1), std::move(1), std::move(seed));
}

EQGenerator::EQGenerator(std::string model_name, std::string faulting,
                         std::string simulation_type, double moment_magnitude,
                         double depth_to_rupt, double rupture_dist, double vs30,
                         double s_or_d, double theta_or_phi, bool truncate) {
  std::string fault_type;
  if (faulting == "StrikeSlip") {
    fault_type = stochastic::FaultType::StrikeSlip;
  } else if (faulting == "ReverseAndRevObliq") {
    fault_type = stochastic::FaultType::ReverseAndRevObliq;
  } else {
    throw std::invalid_argument(
        "ERROR: In EQGenerator::EQGenerator: Input fault type is not correct "
        "or not supported, please check inputs\n");
  }

  std::string sim_type;
  if (simulation_type == "PulseAndNoPulse") {
    sim_type = stochastic::SimulationType::PulseAndNoPulse;
  } else if (simulation_type == "Pulse") {
    sim_type = stochastic::SimulationType::Pulse;
  } else if (simulation_type == "NoPulse") {
    sim_type = stochastic::SimulationType::NoPulse;
  } else {
    throw std::invalid_argument("ERROR: In EQGenerator::EQGenerator: Input "
                                "simulation type is not correct "
                                "or not supported, please check inputs\n");
  }

  eq_model_ =
      Factory<stochastic::StochasticModel, stochastic::DabaghiDerKiureghian,
              stochastic::FaultType, stochastic::SimulationType, double, double,
              double, double, double, double, unsigned int, unsigned int,
              bool>::instance()
          ->create(model_name, std::move(faulting), std::move(sim_type),
                   std::move(moment_magnitude), std::move(depth_to_rupt),
                   std::move(rupture_dist), std::move(vs30), std::move(s_or_d),
                   std::move(theta_or_phi), std::move(1), std
                   : move(1), std::move(truncate));
}

EQGenerator::EQGenerator(std::string model_name, std::string faulting,
                         std::string simulation_type, double moment_magnitude,
                         double depth_to_rupt, double rupture_dist, double vs30,
                         double s_or_d, double theta_or_phi, bool truncate,
                         int seed) {
  std::string fault_type;
  if (faulting == "StrikeSlip") {
    fault_type = stochastic::FaultType::StrikeSlip;
  } else if (faulting == "ReverseAndRevObliq") {
    fault_type = stochastic::FaultType::ReverseAndRevObliq;
  } else {
    throw std::invalid_argument(
        "ERROR: In EQGenerator::EQGenerator: Input fault type is not correct "
        "or not supported, please check inputs\n");
  }

  std::string sim_type;
  if (simulation_type == "PulseAndNoPulse") {
    sim_type = stochastic::SimulationType::PulseAndNoPulse;
  } else if (simulation_type == "Pulse") {
    sim_type = stochastic::SimulationType::Pulse;
  } else if (simulation_type == "NoPulse") {
    sim_type = stochastic::SimulationType::NoPulse;
  } else {
    throw std::invalid_argument("ERROR: In EQGenerator::EQGenerator: Input "
                                "simulation type is not correct "
                                "or not supported, please check inputs\n");
  }

  eq_model_ =
      Factory<stochastic::StochasticModel, stochastic::DabaghiDerKiureghian,
              stochastic::FaultType, stochastic::SimulationType, double, double,
              double, double, double, double, unsigned int, unsigned int,
              bool, int>::instance()
          ->create(model_name, std::move(faulting), std::move(sim_type),
                   std::move(moment_magnitude), std::move(depth_to_rupt),
                   std::move(rupture_dist), std::move(vs30), std::move(s_or_d),
                   std::move(theta_or_phi), std::move(1), std
                   : move(1), std::move(truncate), std::move(seed));
}

utilities::JsonObject EQGenerator::generate_time_history(
    const std::string& event_name) {
  return eq_model_->generate(event_name, true);
}
