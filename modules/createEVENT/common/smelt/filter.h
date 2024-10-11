#ifndef _FILTER_H_
#define _FILTER_H_

#include <functional>
#include <vector>

/**
 * Signal processing functionality
 */
namespace signal_processing {

/**
 * Function for calculating the coefficients of the highpass Butterworth filter
 * @param[in] filter_order Order of the Butterworth filter
 * @param[in] cuttoff_freq Normalized cutoff frequency
 * @return Returns a vector containing two vectors where the first vector
 *         contains and numerator coefficients and the second vector contains
 *         the denominator coefficients.
 */
std::function<std::vector<std::vector<double>>(int, double)> hp_butterworth();

/**
 * Function that calculates the impulse response of a filter defined by the
 * input numerator and denominator coefficients for the input number of samples
 * @param[in] numerator_coeffs Numerator coefficients for filter
 * @param[in] denominator_coeffs Denominator coefficients for filter
 * @param[in] order Order of the filter
 * @param[in] num_samples Number of samples desired
 * @return Vector containing impulse response for requested number of samples
 */
std::function<std::vector<double>(std::vector<double>, std::vector<double>, int,
                                  int)>
    impulse_response();

/**
 * Function that calculates the acausal Butterworth filter for the requested
 * number of samples at input cutoff frequency
 * @param[in] freq_corner Corner frequency
 * @param[in] time_step Time step between observations
 * @param[in] order Order of the filter
 * @param[in] num_samples Number of samples desired
 * @return Vector containing filter coefficients for requested number of samples
 */
std::function<std::vector<double>(double, double, unsigned int, unsigned int)>
    acausal_highpass_filter();
}  // namespace signal_processing

#endif  // _FILTER_H_
