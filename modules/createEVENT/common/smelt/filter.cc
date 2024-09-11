#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
// #include <ipps.h>
#include <fstream>
#include <iostream>
#include <iomanip>  // For std::setprecision


// Eigen dense matrices
#include <Eigen/Dense>

namespace signal_processing {

std::function<std::vector<std::vector<double>>(int, double)> hp_butterworth() {
  return [](int filter_order,
            double cutoff_freq) -> std::vector<std::vector<double>> {

    // // Allocated memory for coefficients
    // std::vector<Ipp64f> taps(2 * (filter_order + 1));
    // IppStatus status = ippStsNoErr;
    // int internal_buffer_size;

    // // Calculate required buffer size for internal calculations
    // status = ippsIIRGenGetBufferSize(filter_order, &internal_buffer_size);
    // if (status != ippStsNoErr) {
    //   throw std::runtime_error(
    //       "\nERROR: in signal_processing::hp_butterworth: Error in buffer size "
    //       "calculations\n");
    // }

    // // Divide by 2 to make cutoff frequency match the definition given in MATLAB
    // Ipp8u* internal_calcs = ippsMalloc_8u(internal_buffer_size);
    // status =
    //     ippsIIRGenHighpass_64f(cutoff_freq / 2.0, 0, filter_order, taps.data(),
    //                            ippButterworth, internal_calcs);

    // // Check if filter computation succeeded
    // if (status != ippStsNoErr) {
    //   throw std::runtime_error(
    //       "\nERROR: in signal_processing::hp_butterworth: Error in coefficient "
    //       "calculations\n");
    // }

    // std::vector<double> numerator(filter_order + 1);
    // std::vector<double> denominator(filter_order + 1);

    // for (int i = 0; i < filter_order + 1; ++i) {
    //   numerator[i] = taps[i];
    //   denominator[i] = taps[i + filter_order + 1];
    // }

    // // Free memory associated with internal calcs
    // ippsFree(internal_calcs);


    if ((filter_order==4)&&(cutoff_freq==0.004)) {
      // all good
    } else {
      throw std::runtime_error("We only support filter_order==4 and cutoff_freq==0.004. Other setups require manual computation of butterworth highpass filter parameters by changing the source code");
    }
   
    std::vector<double> numerator{ 0.983715174129757,-3.934860696519026,5.902291044778539,-3.934860696519026,0.983715174129757 };
    std::vector<double> denominator{ 1.000000000000000, -3.967162595948848, 5.902025861490879, -3.902558784823240, 0.967695543813137 };

    return std::vector<std::vector<double>>{numerator, denominator};
  };
}

std::function<std::vector<double>(std::vector<double>, std::vector<double>, int,
                                  int)>
    impulse_response() {

  return [](const std::vector<double>& numerator_coeffs,
            const std::vector<double>& denominator_coeffs, int order,
            int num_samples) -> std::vector<double> {

  std::cout << "impulse_response needs update" << std::endl;
    if (numerator_coeffs.size() != denominator_coeffs.size()) {
      throw std::runtime_error(
          "\nERROR: in signal_processing::impulse_response: Inputs for "
          "numerator "
          "and denominator coefficients not same length\n");
    }

    // IppStatus status = ippStsNoErr;
    // int internal_buffer_size;
    // IppsIIRState_64f* filter_state = nullptr;
    // Ipp64f *samples = ippsMalloc_64f(num_samples),
    //        *impulse = ippsMalloc_64f(num_samples);
    // std::vector<double> taps(numerator_coeffs.size() +
    //                          denominator_coeffs.size());

    // // Set all values to zero except first one for impulse
    // impulse[0] = 1.0;
    // for (int i = 1; i < num_samples; ++i) {
    //   impulse[i] = 0.0;
    // }

    // // Put filter coefficients into single stack array
    // for (unsigned int i = 0; i < numerator_coeffs.size(); ++i) {
    //   taps[i] = numerator_coeffs[i];
    //   taps[i + numerator_coeffs.size()] = denominator_coeffs[i];
    // }

    // // Get buffer size required for internal calcs
    // status = ippsIIRGetStateSize_64f(order, &internal_buffer_size);
    // if (status != ippStsNoErr) {
    //   throw std::runtime_error(
    //       "\nERROR: in signal_processing::impulse_response: Error in buffer "
    //       "size "
    //       "calculations\n");
    // }

    // // Allocate memory for internal calcs
    // Ipp8u* internal_calcs = ippsMalloc_8u(internal_buffer_size);

    // // Initialize filter state
    // status = ippsIIRInit_64f(&filter_state, taps.data(), order, nullptr,
    //                          internal_calcs);
    // if (status != ippStsNoErr) {
    //   throw std::runtime_error(
    //       "\nERROR: in signal_processing::impulse_response: Error in filter "
    //       "initialization\n");
    // }

    // // Apply filter to impulse
    // status = ippsIIR_64f(impulse, samples, num_samples, filter_state);
    // if (status != ippStsNoErr) {
    //   throw std::runtime_error(
    //       "\nERROR: in signal_processing::impulse_response: Error in filter "
    //       "application\n");
    // }

    // std::vector<double> sample_vec(num_samples);
    // for (int i = 0; i < num_samples; ++i) {
    //   sample_vec[i] = samples[i];
    // }

    // // Free memory used for filtering
    // ippsFree(samples);
    // ippsFree(impulse);
    // ippsFree(internal_calcs);


    // Initialize impulse and samples vectors
    std::vector<double> impulse(num_samples, 0.0);
    impulse[0] = 1.0;  // Set first value to 1 for impulse
    std::vector<double> samples(num_samples, 0.0);

    // Convert the coefficient vectors to Eigen objects
    Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(numerator_coeffs.data(), numerator_coeffs.size());
    Eigen::VectorXd a = Eigen::Map<const Eigen::VectorXd>(denominator_coeffs.data(), denominator_coeffs.size());

    if (a[0] == 0) {
      throw std::runtime_error("Denominator's first coefficient (a[0]) cannot be zero");
    }

    // Normalize the coefficients
    b /= a[0];
    a /= a[0];

    // Apply the filter (difference equation)
    for (int n = 0; n < num_samples; ++n) {
      samples[n] = b[0] * impulse[n];
      for (int i = 1; i <= order; ++i) {
        if (n - i >= 0) {
          samples[n] += b[i] * impulse[n - i] - a[i] * samples[n - i];
        }
      }
    }
    
    std::vector<double> sample_vec(num_samples);
    for (int i = 0; i < num_samples; ++i) {
      sample_vec[i] = samples[i];
    }
    return sample_vec;
  };
}

std::function<std::vector<double>(double, double, unsigned int, unsigned int)>
    acausal_highpass_filter() {
  return [](double freq_corner, double time_step, unsigned int order,
            unsigned int num_samples) -> std::vector<double> {
    
    // Calculate normalized frequency
    double freq_cutoff_norm = 1.0 / (2.0 * time_step);
    
    // Initialize filter and frequencies
    Eigen::VectorXd filter(num_samples);
    std::vector<double> freq_steps(static_cast<unsigned int>(num_samples / 2) +
                                   1);
    double step_freq = freq_cutoff_norm / static_cast<double>(num_samples / 2);

    // Create vector of frequencies ranging from 0 to normalized cutoff
    // frequency
    for (unsigned int i = 0; i < freq_steps.size(); ++i) {
      freq_steps[i] = static_cast<double>(i) * step_freq;
    }

    // Calculate first half of filter coefficients
    for (unsigned int i = 0; i < freq_steps.size(); ++i) {
      filter(i) =
          std::sqrt(1.0 / (1.0 + std::pow(freq_corner / freq_steps[i],
                                          2.0 * order)));
    }

    // Mirror coefficients
    Eigen::VectorXd highpass_filter(2 * filter.size() - 2);
    highpass_filter.head(filter.size()) = filter;
    highpass_filter.segment(filter.size(), filter.size() - 2) =
        filter.segment(1, filter.size() - 2).reverse();

    // Place filter coefficients in STL vector
    std::vector<double> filter_vector(highpass_filter.size());
    Eigen::VectorXd::Map(&filter_vector[0], highpass_filter.size()) =
        highpass_filter;

    return filter_vector;
  };
}
}  // namespace signal_processing
