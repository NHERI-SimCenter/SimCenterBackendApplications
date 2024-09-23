#ifndef _WIND_PROFILE_H_
#define _WIND_PROFILE_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace wind {

std::function<double(const std::string&, const std::vector<double>&, double,
                     double, std::vector<double>&)>
    exposure_category_velocity() {
  return [](const std::string& exposure_category,
            const std::vector<double>& heights, double karman_const,
            double gust_speed, std::vector<double>& velocity_prof) -> double {
    double roughness_ht = 0.0, power_factor = 0.0, power_exponent = 0.0;

    // Set parameters based on exposure category
    if (exposure_category == "A") {
      roughness_ht = 2.0;
      power_factor = 0.3;
      power_exponent = 1.0 / 3.0;
    } else if (exposure_category == "B") {
      roughness_ht = 0.3;
      power_factor = 0.45;
      power_exponent = 1.0 / 4.0;
    } else if (exposure_category == "C") {
      roughness_ht = 0.02;
      power_factor = 0.65;
      power_exponent = 1.0 / 6.5;
    } else if (exposure_category == "D") {
      roughness_ht = 0.005;
      power_factor = 0.80;
      power_exponent = 1.0 / 9.0;
    } else {
      throw std::invalid_argument(
          "\nERROR: In wind::exposure_category_velocity function: Input "
          "exposure "
          "category is not valid, please check input value\n");
    }

    velocity_prof.resize(heights.size());

    for (unsigned int i = 0; i < heights.size(); ++i) {
      velocity_prof[i] = gust_speed * power_factor *
                         std::pow(heights[i] / 10.0, power_exponent);
    }

    return gust_speed * power_factor * karman_const /
           std::log(10.0 / roughness_ht);
  };
}
}  // namespace wind

#endif  // _WIND_PROFILE_H_
