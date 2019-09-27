#ifndef _EQ_GENERATOR_H_
#define _EQ_GENERATOR_H_

#include <string>
#include <vector>
#include <json_object.h>
#include <stochastic_model.h>

/**
 * Class for generating time histories for particular earthquake event
 */
class EQGenerator {
 public:
  /**
   * @constructor Delete default constructor
   */
  EQGenerator() = delete;

  /**
   * @constructor Construct earthquake generator for earthquake at input site
   * location with input moment magnitude
   * @param[in] model_name Name of stochastic model
   * @param[in] moment_magnitude Moment magnitude for earthquake event
   * @param[in] rupture_dist Closest-to-site rupture distance in kilometers
   * @param[in] vs30 Soil shear wave velocity averaged over top 30 meters in
   *                 meters per second
   */
  EQGenerator(std::string model_name, double moment_magnitude, double rupture_dist, double vs30);
  
  /**
   * @constructor Construct earthquake generator for earthquake at input site
   * location with input moment magnitude with specified seed
   * @param[in] model_name Name of stochastic model
   * @param[in] moment_magnitude Moment magnitude for earthquake event
   * @param[in] rupture_dist Closest-to-site rupture distance in kilometers
   * @param[in] vs30 Soil shear wave velocity averaged over top 30 meters in
   *                 meters per second
   * @param[in] seed Value to seed random generator with
   */
  EQGenerator(std::string model_name, double moment_magnitude,
              double rupture_dist, double vs30, int seed);


  /**
   * @constructor Construct earthquake generator for earthquake with input characteristics
   * @param[in] model_name Name of stochastic model
   * @param[in] faulting Type of faulting
   * @param[in] simulation_type Pulse-type of simulation
   * @param[in] moment_magnitude Moment magnitude for earthquake event
   * @param[in] depth_to_rupt Depth to the top of the rupture plane in kilometers
   * @param[in] rupture_dist Closest-to-site rupture distance in kilometers
   * @param[in] vs30 Soil shear wave velocity averaged over top 30 meters in
   *                 meters per second
   * @param[in] s_or_d Directivity parameter s or d (km)--input the larger of
   *                   the two
   * @param[in] theta_or_phi Directivity ange parameter theta or phi
   *                         (degrees)--input corresponding value to s or d
   * @param[in] truncate Boolean indicating whether to truncate and baseline correct
   *                     synthetic motion. 
   */
  EQGenerator(std::string model_name, std::string faulting,
              std::string simulation_type, double moment_magnitude,
              double depth_to_rupt, double rupture_dist, double vs30,
              double s_or_d, double theta_or_phi, bool truncate);

  /**
   * @constructor Construct earthquake generator for earthquake with input characteristics
   * and provided seed
   * @param[in] model_name Name of stochastic model
   * @param[in] faulting Type of faulting
   * @param[in] simulation_type Pulse-type of simulation
   * @param[in] moment_magnitude Moment magnitude for earthquake event
   * @param[in] depth_to_rupt Depth to the top of the rupture plane in kilometers
   * @param[in] rupture_dist Closest-to-site rupture distance in kilometers
   * @param[in] vs30 Soil shear wave velocity averaged over top 30 meters in
   *                 meters per second
   * @param[in] s_or_d Directivity parameter s or d (km)--input the larger of
   *                   the two
   * @param[in] theta_or_phi Directivity ange parameter theta or phi
   *                         (degrees)--input corresponding value to s or d
   * @param[in] truncate Boolean indicating whether to truncate and baseline correct
   *                     synthetic motion.
   * @param[in] seed Value to seed random generator with
   */
  EQGenerator(std::string model_name, std::string faulting,
              std::string simulation_type, double moment_magnitude,
              double depth_to_rupt, double rupture_dist, double vs30,
              double s_or_d, double theta_or_phi, bool truncate, int seed);

  /**
   * Generate time history based on model parameters
   * @param[in] event_name Name to use for event
   * @return JsonObject containing loading time histories
   */
  utilities::JsonObject generate_time_history(const std::string& event_name);

 private:
  std::shared_ptr<stochastic::StochasticModel> eq_model_; /**< Stochastic model
                                                             to generate
                                                             earthquake time
                                                             histories */
};

#endif // _EQ_GENERATOR_H_
