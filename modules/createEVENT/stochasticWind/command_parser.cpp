#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <clara.hpp>
#include "command_parser.h"

CommandParser::CommandParser(int& number_of_arguments, char* arguments[]) {
  command_parser_ =
      clara::detail::Help(configuration_.help).optional() |
      clara::detail::Opt(configuration_.event_file,
                         "Event file location")["--filenameEVENT"](
          "Location where generated time history should be stored")
          .required() |
      clara::detail::Opt(configuration_.bim_file,
                         "BIM file location")["--filenameAIM"](
          "Location where building information model is stored")
          .required() |
      clara::detail::Opt(configuration_.seed, "Seed value")["--seed"](
          "Seed value that should be used to generate time histories")
          .required() |
      clara::detail::Opt(configuration_.rv_flag)["-r"]["--getRV"](
          "Flag indicating whether the generated event file should specify "
          "random variable")
          .optional();

  auto result = command_parser_.parse(clara::detail::Args(number_of_arguments, arguments));
  
  // Check whether command line was succussfully parsed
  if (!result) {
    std::cerr << "ERROR: In command line inputs: " << result.errorMessage() << std::endl;
    throw std::invalid_argument("ERROR: In CommandParser::CommandParser: Command line inputs error");
  }

  // If help flag passed, pring usage and exit
  if (configuration_.help) {
    std::cout << command_parser_ << std::endl;
  }
}

bool CommandParser::seed_provided() const {
  if (configuration_.seed == "None") {
    return false;
  } else {
    return true;
  }  
}

int CommandParser::get_seed() const {
  int seed_value = 0;
  try {
    seed_value = std::stoi(configuration_.seed);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: In CommandParser::get_seed(): Invalid string to int "
                 "conversion. Check input seed value: "
              << e.what() << std::endl;
  }
  return seed_value;
}

std::string CommandParser::get_event_file() const {
  return configuration_.event_file;
}

std::string CommandParser::get_bim_file() const {
  return configuration_.bim_file;
}

bool CommandParser::get_help_flag() const {
  return configuration_.help;
}

bool CommandParser::get_rv_flag() const {
  return configuration_.rv_flag;
}

CommandParser::Config CommandParser::get_configuration() const {
  return configuration_;
}
