#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <common/Units.h>
#include <configure.h>
#include <nlohmann/json.hpp>
#include "command_parser.h"
#include "function_dispatcher.h"
#include "wind_generator.h"
#include <filesystem>

using json = nlohmann::json;
typedef std::chrono::duration<
    int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8> >::type>
    Days; /* UTC: +8:00 */

int main(int argc, char** argv) {

  // Read command line arguments
  CommandParser inputs;
  try {
    inputs = CommandParser(argc, argv);    
  } catch (const std::invalid_argument& e) {
    std::cerr << "\nException caught in command line processing: " << e.what() << std::endl;
    return 1;
  }
  
  // Check if help menu requested
  if (inputs.get_help_flag()) {
    return 0;
  }

  // Initialize stochastic load generation library
  config::initialize();

  // Get the building information, units
  Units::UnitSystem bim_units, event_units;
  std::ifstream bim_file(inputs.get_bim_file());
  json input_data;
  bim_file >> input_data;

  bim_units.lengthUnit =
      Units::ParseLengthUnit(input_data["GeneralInformation"]["units"]["length"]
                                 .get<std::string>()
                                 .c_str());
  bim_units.forceUnit =
      Units::ParseForceUnit(input_data["GeneralInformation"]["units"]["force"]
                                 .get<std::string>()
                                 .c_str());
  event_units.lengthUnit = Units::ParseLengthUnit("m");
  event_units.forceUnit = Units::ParseForceUnit("N");

  double length_conversion = Units::GetLengthFactor(bim_units, event_units);
  double force_conversion = Units::GetForceFactor(event_units, bim_units);
  double height = input_data["GeneralInformation"]["height"].get<double>() *
                  length_conversion;
  double width = input_data["GeneralInformation"]["width"].get<double>() *
                 length_conversion;
  int num_floors = input_data["GeneralInformation"]["stories"];
  // Loop over input events and generate corresponding time histories
  json events_array = json::array();

  try {
    if (num_floors <= 0 || height <= 0.0 || width <= 0.0) {
      throw std::runtime_error(
          "ERROR: In main() of StochasticWindGenerator: Check General "
          "Information inputs. Number of floors, height and width all must be "
          "greater than 0\n");
      return 1;
    }

    for (json::iterator it = input_data["Events"].begin();
         it != input_data["Events"].end(); ++it) {
      if (it->at("type") == "StochasticWindWittigSinha") {
        json current_event;
        current_event.emplace("type", "Wind");
        current_event.emplace("subtype", "StochasticWindWittigSinha");

	std::string exposure_category = it->at("exposureCategory");
        double drag_coeff = 1.0, gust_wind_speed = 100.0;
        double total_time = 600.0; // This generates 10 mins of wind loading,
                                   // based on provided MATLAB code

        // Random variable flag has NOT been passed
        if (!inputs.get_rv_flag()) {
          drag_coeff = it->at("dragCoefficient");
          gust_wind_speed = it->at("gustSpeed");
          gust_wind_speed = gust_wind_speed * 0.44704; // Convert from mph to m/s

	  if (drag_coeff <= 0.0 || gust_wind_speed <= 0.0) {
            throw std::runtime_error(
                "ERROR: In main() of StochasticWindGenerator: Check drag "
                "coefficient and gust wind speed; both must be "
                "greater than 0.0\n");
          }
        }

        // Generate wind loading based on parameters provided
        // Calculate number of nanoseconds in time to use as seed
        std::chrono::time_point<std::chrono::system_clock> now =
            std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        Days days = std::chrono::duration_cast<Days>(duration);
        duration -= days;
        auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
        auto minutes =
            std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
        auto seconds =
            std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;
        auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        duration -= milliseconds;
        auto microseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(duration);
        duration -= microseconds;
        auto nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration);


        // main seed
        int seed;
        if (input_data["Events"][0]["seed"].is_string()) {
            // if this is "None"
            seed = nanoseconds.count();

        } else {

            int base_seed = input_data["Events"][0]["seed"].get<int>();

            // Get the current working directory
            std::string folderName = std::filesystem::path(std::filesystem::current_path()).filename().string();

            
            // Split by '.' and get the last part (sampNum)
            std::size_t pos = folderName.find_last_of('.');

            if (pos!= std::string::npos) {
                           std::string samp_num_str = (pos != std::string::npos) ? folderName.substr(pos + 1) : folderName;
                            int samp_num = std::stoi(samp_num_str);
                            seed = samp_num + base_seed; 
                        } else {
                            seed = base_seed; 
                        }


         }


        std::cout << seed << std::endl;
            auto wind_forces = Dispatcher<std::tuple<std::vector<double>, json>, std::string,
                                 double, double, double, double, unsigned int,
                                 double, double, int>::instance()
                          ->dispatch("WittigSinha1975", exposure_category,
                                     gust_wind_speed, drag_coeff, width, height,
                                     num_floors, total_time, force_conversion,
                                     seed);

            // auto wind_forces =
            //     inputs.seed_provided()
            //         ? Dispatcher<std::tuple<std::vector<double>, json>, std::string,
            //                      double, double, double, double, unsigned int,
            //                      double, double, int>::instance()
            //               ->dispatch("WittigSinha1975", exposure_category,
            //                          gust_wind_speed, drag_coeff, width, height,
            //                          num_floors, total_time, force_conversion,
            //                          inputs.get_seed())
            //         : Dispatcher<std::tuple<std::vector<double>, json>, std::string,
            //                      double, double, double, double, unsigned int,
            //                      double, double, int>::instance()
            //               ->dispatch("WittigSinha1975", exposure_category,
            //                          gust_wind_speed, drag_coeff, width, height,
            //                          num_floors, total_time, force_conversion,
            //                          nanoseconds.count());


    auto static_forces = std::get<0>(wind_forces);
	auto dynamic_forces = std::get<1>(wind_forces);
	auto pattern = json::array();
	auto time_series = json::array();

	// Create pattern for each floor and add static floor loads
	for (unsigned int i = 0; i < static_forces.size(); ++i) {
          auto floor_pattern =
              json::object({{"dof", 1},
                            {"floor", std::to_string(i + 1)},
                            {"name", std::to_string(i + 1)},
                            {"timeSeries", std::to_string(i + 1)},
                            {"staticWindLoad", static_forces[i]},
                            {"type", "WindFloorLoad"}});
          pattern.push_back(floor_pattern);
	}
	
	// Add pattern and time series to current event
        current_event.emplace("pattern", pattern);
        current_event.emplace("timeSeries",
                              dynamic_forces["Events"][0]["timeSeries"]);
        current_event.emplace(
            "dT", dynamic_forces["Events"][0]["timeSeries"][0]["dT"]);
        current_event.emplace(
            "numSteps",
            dynamic_forces["Events"][0]["timeSeries"][0]["data"].size());

        // Add event to current events array
	events_array.push_back(current_event);

      } else {
        events_array.push_back(*it);
      }
    }
  } catch (const std::exception &e) {
    std::cerr
        << "ERROR: In main() of StochasticWindGenerator: "
        << e.what() << std::endl;
    return 1;
  }

  // Add events array to event JSON object and write to file
  json event;
  event.emplace("Events", events_array);

  // Write prettyfied JSON to file
  std::ofstream event_file;
  event_file.open(inputs.get_event_file());

  if (!event_file.is_open()) {
    std::cerr << "\nERROR: In main() of StochasticWindGenerator: Could "
                 "not open output location\n";
  }
  event_file << std::setw(4) << event << std::endl;
  event_file.close();

  if (event_file.fail()) {
    std::cerr << "\nERROR: In In main() of StochasticWindGenerator:: Error when "
                 "closing output location\n";
  }

  return 0;
}
