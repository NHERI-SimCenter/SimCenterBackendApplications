#ifndef _WIND_PROFILE_H_
#define _WIND_PROFILE_H_

#define _USE_MATH_DEFINES
#include <functional>
#include <string>
#include <vector>

/**
 * Wind profile generation functionality
 */
namespace wind {

/**
 * Function that calculated the vertical wind velocity and friction velocity
 * profiles based on ASCE exposure categories using power law description for
 * wind velocity
 * @param[in] exposure_cat ASCE exposure category
 * @param[in] heights Vector of heights at which to calculate velocity
 * @param[in] karman_const Value to use for Von Karman constant
 * @param[in] gust_speed Gust wind speed
 * @param[in, out] velocity_prof Vector to store vertical velocity profile in
 * @return Value of the friction velocity
 */
std::function<double(const std::string&, const std::vector<double>&, double,
                     double, std::vector<double>&)>
    exposure_category_velocity();
}  // namespace wind

#endif  // _WIND_PROFILE_H_
