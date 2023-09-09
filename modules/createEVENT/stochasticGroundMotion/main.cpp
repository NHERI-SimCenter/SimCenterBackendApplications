#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <iomanip>
#include <memory>
#include <stdexcept>
#include <configure.h>
#include <nlohmann/json.hpp>
#include "command_parser.h"
#include "eq_generator.h"

using json = nlohmann::json;
typedef std::chrono::duration<
    int, std::ratio_multiply<std::chrono::hours::period, std::ratio<8> >::type>
    Days; /* UTC: +8:00 */

void throwError(std::string msg);

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
  
  // Check if random variable flag has been passed
  if (inputs.get_rv_flag()) {

    json event;

    //
    // SY - adding seed random variable
    //

    //json rv_seed_description;
    // rv_seed_description.emplace("distribution", "Uniform");
    // rv_seed_description.emplace("name", "StochasticSeed");
    // rv_seed_description.emplace("value", "RV.StochasticSeed");
    // rv_seed_description.emplace("inputType", "Parameters");
    // rv_seed_description.emplace("lowerbound", 0);
    // rv_seed_description.emplace("upperbound", 1);
    // rv_seed_description.emplace("variableClass", "Uncertain");
    auto rv_seed_description = json::object({{"distribution", "Uniform"},
                                 {"name", "StochasticSeed"},
                                 {"value", "RV.StochasticSeed"},
                                 {"inputType", "Parameters"},
                                 {"refCount", 1},
                                 {"lowerbound", 0},
                                 {"upperbound", 1},
                                 {"variableClass", "Uncertain"}});


    event.emplace("randomVariables", json::array({rv_seed_description}));

    json event_description;
    event_description.emplace("type", "Seismic");
    event_description.emplace("subtype", "StochasticGroundMotion");

    auto pattern = json::object({{"dof", 1},
                                 {"timeSeries", "accel_x"},
                                 {"type", "UniformAcceleration"}});

    event_description.emplace("pattern", json::array({pattern}));

    event.emplace("Events", json::array({event_description}));
    std::ofstream event_file;
    event_file.open(inputs.get_event_file());

    if (!event_file.is_open()) {
      std::cerr << "\nERROR: In main() of StochasticGroundMotion: Could "
                   "not open output location\n";
    }

    // Write prettyfied JSON to file
    event_file << std::setw(4) << event << std::endl;
    event_file.close();

    if (event_file.fail()) {
      throwError( "\nERROR: In In main() of StochasticGroundMotion:: Error when closing output location\n");    
      // std::cerr
      //     << "\nERROR: In In main() of StochasticGroundMotion:: Error when "
      //        "closing output location\n";
    }
  // No random variable flag passed, so generate ground motion
  } else {
    std::shared_ptr<EQGenerator> eq_generator;
    json event;
    event.emplace("randomVariables", json::array());

    std::ifstream bim_file(inputs.get_bim_file());
    json input_data;
    bim_file >> input_data;
    
    for (json::iterator it = input_data["Events"].begin(); it != input_data["Events"].end(); ++it) {
      try {
      	std::string model_name = inputs.get_model_name();

        // find if seed exists
        bool seed_exists = false;

        //
        // Sang-ri - checking if key seed exists in AIM.json. We are not anymore getting the seed
        //           from the commend line.
        //


        auto invIt = *it;
        // Enumerate all keys
        for (auto item : invIt.items())
          {
              if (item.key() == "seed") {
                seed_exists = true;
              }
        }
        //std::cout<< seed_exists << std::endl;

        if (seed_exists){

            double mySeed_tmp = invIt["seed"];
            mySeed_tmp *= 1.e9;
            int mySeed = (int) std::round(mySeed_tmp);

            std::cout<< " seed: " << mySeed << std::endl;
            std::cout<< model_name << std::endl;
            
            // Vlachos et al (2018) model
            if (model_name == "VlachosSiteSpecificEQ") {

                //
                // Protecting the model
                //

                double myMoment = it->at("momentMagnitude");
                double myR = it->at("ruptureDist");
                bool modelProtected = it->at("protectModel");

                if ((myMoment>7.5) && (modelProtected)) {
                  std::string msg = "ERROR: In StochasticGroundMotion:: The moment magnitude greater than 7.5 is not validated, but the provided value is " + std::to_string(myMoment);
                  throwError(msg);
                }

                if ((myR<5) && (modelProtected)) {
                  std::string msg = "ERROR: In StochasticGroundMotion:: The closest-to-site distance smaller than 5 km is not validated, but the provided value is " + std::to_string(myR) + "km";
                  throwError(msg);
                }

              //eq_generator = std::make_shared<EQGenerator>(
              //    inputs.get_model_name(), it->at("momentMagnitude"),
              //    it->at("ruptureDist"), it->at("vs30"), inputs.get_seed());
              eq_generator = std::make_shared<EQGenerator>(
                  inputs.get_model_name(), it->at("momentMagnitude"),
                  it->at("ruptureDist"), it->at("vs30"), mySeed); // May need to update smelt not to cut of seed nubmers
            } else if (model_name == "DabaghiDerKiureghianNFGM") {
              //eq_generator = std::make_shared<EQGenerator>(
              //    inputs.get_model_name(), it->at("faultType"),
              //    it->at("simulationType"), it->at("momentMagnitude"),
              //    it->at("depthToRupt"), it->at("ruptureDist"), it->at("vs30"),
              //    it->at("sOrD"), it->at("thetaOrPhi"), it->at("truncate"),
              //    inputs.get_seed());
              eq_generator = std::make_shared<EQGenerator>(
                 inputs.get_model_name(), it->at("faultType"),
                 it->at("simulationType"), it->at("momentMagnitude"),
                 it->at("depthToRupt"), it->at("ruptureDist"), it->at("vs30"),
                 it->at("sOrD"), it->at("thetaOrPhi"), it->at("truncate"),
                 mySeed);
            } else {
              throwError( "ERROR: In main() of StochasticGroundMotion: Earthquake model name either incorrect or not supported, please check inputs\n");    

              // throw std::runtime_error(
              //     "ERROR: In main() of StochasticGroundMotion: Earthquake model "
              //     "name either incorrect or not supported, please check "
              //     "inputs\n");	    
            }
          } else {
      	  const auto clock_time = std::chrono::time_point<std::chrono::system_clock>{};
      	  const auto current_time = std::chrono::system_clock::now();

      	  const auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - clock_time);

            if (model_name == "VlachosSiteSpecificEQ") {
              eq_generator = std::make_shared<EQGenerator>(
                  inputs.get_model_name(), it->at("momentMagnitude"),
                  it->at("ruptureDist"), it->at("vs30"), time_diff.count());
            } else if (model_name == "DabaghiDerKiureghianNFGM") {
              eq_generator = std::make_shared<EQGenerator>(
                  inputs.get_model_name(), it->at("faultType"),
                  it->at("simulationType"), it->at("momentMagnitude"),
                  it->at("depthToRupt"), it->at("ruptureDist"), it->at("vs30"),
                  it->at("sOrD"), it->at("thetaOrPhi"), it->at("truncate"),
                  time_diff.count());
            } else {

              throwError( "ERROR: In main() of StochasticGroundMotion: Earthquake model name either incorrect or not supported, please check inputs\n");    
              // throw std::runtime_error(
              //     "ERROR: In main() of StochasticGroundMotion: Earthquake model "
              //     "name either incorrect or not supported, please check "
              //     "inputs\n");
            }
          }

        auto time_history =
            eq_generator->generate_time_history("StochasticMotion")
                .get_library_json();

        if (time_history.at("Events").size() != 1) {
          throwError( "ERROR: In main() of StochasticGroundMotion with getRV flag set: Generated events should have length 1\n");

          // throw std::runtime_error(
          //     "ERROR: In main() of StochasticGroundMotion with getRV "
          //     "flag set: Generated events should have length 1\n");
        }
        auto event_data = time_history.at("Events")[0];

        auto array_entry = json::object(
            {{"name", event_data.at("name")},
             {"type", event_data.at("type")},
             {"dT", event_data.at("dT")},
             {"Data", "Time history generated using " +
                          inputs.get_model_name() + " model"},
             {"numSteps", event_data.at("numSteps")},
             {"timeSeries", json::array({event_data.at("timeSeries")[0]})},
             {"pattern", json::array({event_data.at("pattern")[0]})}});

        auto event_array = json::array();
        event_array.push_back(array_entry);
        event.emplace("Events", event_array);
      } catch (const std::exception& e) {
        throwError("ERROR: In main() of StochasticGroundMotion without getRV flag: " + std::string(e.what()) );
        //std::cerr
        //    << "ERROR: In main() of StochasticGroundMotion without getRV flag: "
        //    << e.what() << std::endl;
        return 1;
      }
    }

    // Write prettyfied JSON to file
    std::ofstream event_file;
    event_file.open(inputs.get_event_file());

    if (!event_file.is_open()) {
      throwError("\nERROR: In main() of StochasticGroundMotion: Could not open output location\n");

      //std::cerr << "\nERROR: In main() of StochasticGroundMotion: Could "
      //             "not open output location\n";
    }
    event_file << std::setw(4) << event << std::endl;
    event_file.close();

    if (event_file.fail()) {
      throwError("\nERROR: In In main() of StochasticGroundMotion:: Error when closing output location\n");
      // std::cerr
      //     << "\nERROR: In In main() of StochasticGroundMotion:: Error when "
      //        "closing output location\n";
    }
  }
  return 0;
}

void throwError(std::string msg){
  std::cerr << msg;

  //
  // get folder id and decide error file name e.g. workflow.err.1
  //

  std::filesystem::path cwd = std::filesystem::current_path(); 
  std::string filePathString = cwd.string();
  std::size_t loc_dot = filePathString.find_last_of(".");
  int id =  std::stoi(filePathString.substr(loc_dot+1)) ;
  std::string errFile = "../workflow.err." + std::to_string(id); // e.g. workflow.err.1

  //
  // Write msg to the file
  //

  std::ofstream myfile;
  myfile.open(errFile);
  myfile << msg;
  myfile.close();

  //
  // exit
  //

  exit(-1);

  return;
}