#include <complex>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <kiss_fft.h>
#include <fstream>

//#include <mkl.h>
//#include <mkl_dfti.h>
//#include <mkl_vsl.h>
#include "numeric_utils.h"

namespace numeric_utils {
Eigen::MatrixXd corr_to_cov(const Eigen::MatrixXd& corr,
                            const Eigen::VectorXd& std_dev) {
  Eigen::MatrixXd cov_matrix = Eigen::MatrixXd::Zero(corr.rows(), corr.cols());

  for (unsigned int i = 0; i < cov_matrix.rows(); ++i) {
    for (unsigned int j = 0; j < cov_matrix.cols(); ++j) {
      cov_matrix(i, j) = corr(i, j) * std_dev(i) * std_dev(j);
    }
  }

  return cov_matrix;
}
  
bool convolve_1d(const std::vector<double>& input_x,
                 const std::vector<double>& input_y,
                 std::vector<double>& response) {


  // bool status = true;
  // response.resize(input_x.size() + input_y.size() - 1);

  // // Create convolution status and task pointer
  // int conv_status;
  // VSLConvTaskPtr conv_task;
  // // Construct convolution task, with solution mode set to direct
  // conv_status =
  //     vsldConvNewTask1D(&conv_task, VSL_CONV_MODE_DIRECT, input_x.size(),
  //                       input_y.size(), response.size());

  // // Check if convolution construction was successful
  // if (conv_status != VSL_STATUS_OK) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::convolve_1d: Error in convolution "
  //       "construction\n");
  //   status = false;
  // }

  // // Set convolution to start at first element in input_y
  // vslConvSetStart(conv_task, 0);

  // // Execute convolution
  // conv_status = vsldConvExec1D(conv_task, input_x.data(), 1, input_y.data(), 1,
  //                              response.data(), 1);

  // // Check if convolution exectution was successful
  // if (conv_status != VSL_STATUS_OK) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::convolve_1d: Error in convolution "
  //       "execution\n");
  //   status = false;
  // }

  // // Delete convolution task
  // vslConvDeleteTask(&conv_task);

    int n_x = input_x.size();
    int n_y = input_y.size();
    int n_response = n_x + n_y - 1;

    // Resize response to hold the convolution result
    response.resize(n_response);

    // Find the next power of 2 to perform FFT (for efficiency)
    int n_fft = 1;
    while (n_fft < n_response) {
        n_fft <<= 1;  // Multiply by 2
    }

    // Allocate FFT configurations for forward and inverse FFT
    kiss_fft_cfg cfg_forward = kiss_fft_alloc(n_fft, 0, nullptr, nullptr);  // 0 means forward FFT
    kiss_fft_cfg cfg_inverse = kiss_fft_alloc(n_fft, 1, nullptr, nullptr);  // 1 means inverse FFT

    if (!cfg_forward || !cfg_inverse) {
        std::cerr << "ERROR: Failed to allocate KissFFT configuration." << std::endl;
        return false;
    }

    // Prepare complex input arrays for FFT
    std::vector<kiss_fft_cpx> input_x_fft(n_fft, {0, 0});
    std::vector<kiss_fft_cpx> input_y_fft(n_fft, {0, 0});

    // Copy real inputs to complex arrays (imaginary part is 0)
    for (int i = 0; i < n_x; ++i) {
        input_x_fft[i].r = input_x[i];
        input_x_fft[i].i = 0.0;
    }
    for (int i = 0; i < n_y; ++i) {
        input_y_fft[i].r = input_y[i];
        input_y_fft[i].i = 0.0;
    }

    // Perform forward FFT on both inputs
    std::vector<kiss_fft_cpx> x_fft_result(n_fft);
    std::vector<kiss_fft_cpx> y_fft_result(n_fft);

    kiss_fft(cfg_forward, input_x_fft.data(), x_fft_result.data());
    kiss_fft(cfg_forward, input_y_fft.data(), y_fft_result.data());

    // Multiply the FFT results element-wise (complex multiplication)
    std::vector<kiss_fft_cpx> convolution_fft(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        convolution_fft[i].r = x_fft_result[i].r * y_fft_result[i].r - x_fft_result[i].i * y_fft_result[i].i;
        convolution_fft[i].i = x_fft_result[i].r * y_fft_result[i].i + x_fft_result[i].i * y_fft_result[i].r;
    }

    // Perform inverse FFT to get the convolution result in time domain
    std::vector<kiss_fft_cpx> result_ifft(n_fft);
    kiss_fft(cfg_inverse, convolution_fft.data(), result_ifft.data());

    // Extract real part of the result and scale it by 1/n_fft
    for (int i = 0; i < n_response; ++i) {
        response[i] = result_ifft[i].r / n_fft;
    }

    // Free the KissFFT configurations
    free(cfg_forward);
    free(cfg_inverse);

  return true;
}

bool inverse_fft(std::vector<std::complex<double>> input_vector,
                 std::vector<double>& output_vector) {

  //output_vector.resize(input_vector.size());
  // // Create task descriptor and MKL status
  // DFTI_DESCRIPTOR_HANDLE fft_descriptor;
  // MKL_LONG fft_status;

  // // Allocate the descriptor data structure and initializes it with default
  // // configuration values
  // fft_status = DftiCreateDescriptor(&fft_descriptor, DFTI_DOUBLE, DFTI_REAL, 1,
  //                               input_vector.size());
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::inverse_fft: Error in descriptor creation\n");
  //   return false;
  // }
  
  // // Set configuration value to not do inplace transformation
  // fft_status = DftiSetValue(fft_descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::inverse_fft: Error in setting configuration\n");
  //   return false;
  // }

  // // Set the backward scale factor to be 1 divided by the size of the input vector
  // // to make the backward tranform the inverse of the forward transform
  // fft_status = DftiSetValue(fft_descriptor, DFTI_BACKWARD_SCALE,
  //                           static_cast<double>(1.0 / input_vector.size()));
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::inverse_fft: Error in setting backward "
  //       "scale factor\n");
  //   return false;
  // }  

  // // Perform all initialization for the actual FFT computation
  // fft_status = DftiCommitDescriptor(fft_descriptor);
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::inverse_fft: Error in committing descriptor\n");
  //   return false;
  // }
  
  // // Compute the backward FFT
  // fft_status = DftiComputeBackward(fft_descriptor, input_vector.data(),
  //                                  output_vector.data());
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::inverse_fft: Error in computing backward FFT\n");
  //   return false;
  // }
  
  // // Free the memory allocated for descriptor
  // fft_status = DftiFreeDescriptor(&fft_descriptor);
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::inverse_fft: Error in freeing FFT descriptor\n");
  //   return false;
  // }  

  int n = input_vector.size();

  // Resize output vector to match the input size
  output_vector.resize(n);

  // Create KissFFT configuration for inverse FFT
  kiss_fft_cfg cfg = kiss_fft_alloc(n, 1, nullptr, nullptr); // '1' indicates inverse FFT
  if (!cfg) {
      std::cerr << "ERROR: Failed to allocate KissFFT configuration." << std::endl;
      return false;
  }

  // Convert input vector from std::complex<double> to kiss_fft_cpx
  std::vector<kiss_fft_cpx> input_kissfft(n);
  for (int i = 0; i < n; ++i) {
      input_kissfft[i].r = input_vector[i].real();
      input_kissfft[i].i = input_vector[i].imag();
  }

  // Allocate output for KissFFT result (complex output even for inverse FFT)
  std::vector<kiss_fft_cpx> output_kissfft(n);

  // Perform the inverse FFT
  kiss_fft(cfg, input_kissfft.data(), output_kissfft.data());

  // Free the KissFFT configuration
  free(cfg);

  // Convert complex KissFFT result to real output (we discard the imaginary parts)
  for (int i = 0; i < n; ++i) {
      output_vector[i] = output_kissfft[i].r / n;  // Scaling the result manually
  }



  return true;
}

bool inverse_fft(const Eigen::VectorXcd& input_vector,
                 Eigen::VectorXd& output_vector) {
  // Convert input Eigen vector to std vector
  std::vector<std::complex<double>> input_vals(input_vector.size());
  std::vector<double> outputs(input_vals.size());
  Eigen::VectorXcd::Map(&input_vals[0], input_vector.size()) = input_vector;
 
  try {
    inverse_fft(input_vals, outputs);
  } catch (const std::exception& e) {
    std::cerr << "\nERROR: In numeric_utils::inverse_fft (With Eigen Vectors):"
              << e.what() << std::endl;
  }

  // Convert output from std vector to Eigen vector
  output_vector = Eigen::Map<Eigen::VectorXd>(outputs.data(), outputs.size());

  return true;
}

bool inverse_fft(const Eigen::VectorXcd& input_vector,
                 std::vector<double>& output_vector) {
  // Convert input Eigen vector to std vector
  std::vector<std::complex<double>> input_vals(input_vector.size());
  Eigen::VectorXcd::Map(&input_vals[0], input_vector.size()) = input_vector;
  output_vector.resize(input_vector.size());  
 
  try {
    inverse_fft(input_vals, output_vector);
  } catch (const std::exception& e) {
    std::cerr << "\nERROR: In numeric_utils::inverse_fft (With Eigen Vectors):"
              << e.what() << std::endl;
  }

  return true;  
}

bool fft(std::vector<double> input_vector,
         std::vector<std::complex<double>>& output_vector) {
  // Convert input vector to complex values

  // std::vector<std::complex<double>> input_complex(input_vector.size());
  // std::copy(input_vector.begin(), input_vector.end(), input_complex.begin());
  
  // output_vector.resize(input_vector.size());

  // // Create task descriptor and MKL status
  // DFTI_DESCRIPTOR_HANDLE fft_descriptor;
  // MKL_LONG fft_status;

  // // Allocate the descriptor data structure and initializes it with default
  // // configuration values
  // fft_status = DftiCreateDescriptor(&fft_descriptor, DFTI_DOUBLE, DFTI_COMPLEX,
  //                                   1, input_complex.size());
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::fft: Error in descriptor creation\n");
  //   return false;
  // }
  
  // // Set configuration value to not do inplace transformation
  // fft_status = DftiSetValue(fft_descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::fft: Error in setting configuration\n");
  //   return false;
  // }

  // // Perform all initialization for the actual FFT computation
  // fft_status = DftiCommitDescriptor(fft_descriptor);
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::fft: Error in committing descriptor\n");
  //   return false;
  // }
  
  // // Compute the backward FFT
  // fft_status = DftiComputeForward(fft_descriptor, input_complex.data(),
  //                                 output_vector.data());
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::fft: Error in computing FFT\n");
  //   return false;
  // }
  
  // // Free the memory allocated for descriptor
  // fft_status = DftiFreeDescriptor(&fft_descriptor);
  // if (fft_status != DFTI_NO_ERROR) {
  //   throw std::runtime_error(
  //       "\nERROR: in numeric_utils::fft: Error in freeing FFT descriptor\n");
  //   return false;
  // }  

  size_t size = input_vector.size();
  
  // Convert input vector to kiss_fft_cpx
  std::vector<kiss_fft_cpx> input_complex(size);
  for (size_t i = 0; i < size; ++i) {
      input_complex[i].r = static_cast<kiss_fft_scalar>(input_vector[i]);
      input_complex[i].i = 0;
  }

  // Prepare output vector
  output_vector.resize(size);

  // Initialize KissFFT configuration
  kiss_fft_cfg cfg = kiss_fft_alloc(size, 0, nullptr, nullptr);
  if (!cfg) {
      throw std::runtime_error("ERROR: in fft: Failed to allocate KissFFT configuration");
      return false;
  }

  // Perform the FFT
  std::vector<kiss_fft_cpx> output_complex(size);
  kiss_fft(cfg, input_complex.data(), output_complex.data());

  // Convert the output to std::complex<double>
  for (size_t i = 0; i < size; ++i) {
      output_vector[i] = std::complex<double>(output_complex[i].r, output_complex[i].i);
  }

  // Free KissFFT configuration
  free(cfg);

  return true;
}

bool fft(const Eigen::VectorXd& input_vector, Eigen::VectorXcd& output_vector) {

  // Convert input Eigen vector to std vector
  std::vector<double> input_vals(input_vector.size());
  std::vector<std::complex<double>> outputs(input_vals.size());
  Eigen::VectorXd::Map(&input_vals[0], input_vector.size()) = input_vector;
 
  try {
    fft(input_vals, outputs);
  } catch (const std::exception& e) {
    std::cerr << "\nERROR: In numeric_utils::fft (With Eigen Vectors):"
              << e.what() << std::endl;
  }

  // Convert output from std vector to Eigen vector
  output_vector = Eigen::Map<Eigen::VectorXcd>(outputs.data(), outputs.size());

  return true;
}

bool fft(const Eigen::VectorXd& input_vector,
                 std::vector<std::complex<double>>& output_vector) {
  // Convert input Eigen vector to std vector
  std::vector<double> input_vals(input_vector.size());
  Eigen::VectorXd::Map(&input_vals[0], input_vector.size()) = input_vector;
  output_vector.resize(input_vector.size());  
 
  try {
    fft(input_vals, output_vector);
  } catch (const std::exception& e) {
    std::cerr << "\nERROR: In numeric_utils::fft (With Eigen Vector and STL vector):"
              << e.what() << std::endl;
  }

  return true;  
}  
  
double trapazoid_rule(const std::vector<double>& input_vector, double spacing) {
  double result = (input_vector[0] + input_vector[input_vector.size() - 1]) / 2.0;

  for (unsigned int i = 1; i < input_vector.size() - 1; ++i) {
    result = result + input_vector[i];
  }

  return result * spacing;
}

double trapazoid_rule(const Eigen::VectorXd& input_vector, double spacing) {
  double result = (input_vector[0] + input_vector[input_vector.size() - 1]) / 2.0;

  for (unsigned int i = 1; i < input_vector.size() - 1; ++i) {
    result = result + input_vector[i];
  }

  return result * spacing;
}

Eigen::VectorXd polyfit_intercept(const Eigen::VectorXd& points,
                                       const Eigen::VectorXd& data,
				       double intercept,
                                       unsigned int degree) {

  Eigen::MatrixXd coefficients =
      Eigen::MatrixXd::Zero(points.size(), degree - 1);
  
  for (unsigned int i = 0; i < degree - 1; ++i) {
    coefficients.col(i) = points.array().pow(degree - i);
  }

  // Solve system
  Eigen::VectorXd solution = coefficients.fullPivHouseholderQr().solve(
      (data.array() - intercept).matrix());

  // Set y-intercept to zero
  Eigen::VectorXd poly_fit(solution.size() + 2);

  for (unsigned int i = 0; i < solution.size(); ++i) {
    poly_fit(i) = solution(i);
  }  
  poly_fit(poly_fit.size() - 1) = intercept;
  poly_fit(poly_fit.size() - 2) = 0.0;

  return poly_fit;
}

Eigen::VectorXd polynomial_derivative(const Eigen::VectorXd& coefficients) {
  Eigen::VectorXd derivative(coefficients.size() - 1);

  for (unsigned int i = 0; i < derivative.size(); ++i) {
    derivative(i) = coefficients(i) * (coefficients.size() - 1 - i);
  }

  return derivative;
}

std::vector<double> derivative(const std::vector<double>& coefficients,
                               double constant_factor, bool add_zero) {

  if (add_zero) {
    std::vector<double> derivative(coefficients.size());
    derivative[0] = coefficients[0] * constant_factor;

    for (unsigned int i = 1; i < derivative.size(); ++i) {
      derivative[i] = (coefficients[i] - coefficients[i - 1]) * constant_factor;
    }

    return derivative;
  } else {
    std::vector<double> derivative(coefficients.size() - 1);

    for (unsigned int i = 0; i < derivative.size(); ++i) {
      derivative[i] = (coefficients[i + 1] - coefficients[i]) * constant_factor;
    }

    return derivative;
  }
}

Eigen::VectorXd evaluate_polynomial(const Eigen::VectorXd& coefficients,
                                    const Eigen::VectorXd& points) {
  Eigen::VectorXd evaluations = Eigen::VectorXd::Zero(points.size());

  for (unsigned int i = 0; i < evaluations.size(); ++i) {
    for (unsigned int j = 0; j < coefficients.size(); ++j) {
      evaluations(i) +=
          coefficients(j) * std::pow(points(i), coefficients.size() - 1 - j);
    }
  }

  return evaluations;
}

Eigen::VectorXd evaluate_polynomial(const Eigen::VectorXd& coefficients,
                                    const std::vector<double>& points) {
  Eigen::VectorXd evaluations = Eigen::VectorXd::Zero(points.size());

  for (unsigned int i = 0; i < evaluations.size(); ++i) {
    for (unsigned int j = 0; j < coefficients.size(); ++j) {
      evaluations(i) +=
          coefficients(j) * std::pow(points[i], coefficients.size() - 1 - j);
    }
  }

  return evaluations;
}

std::vector<double> evaluate_polynomial(const std::vector<double>& coefficients,
                                        const std::vector<double>& points) {
  std::vector<double> evaluations(points.size(), 0.0);

  for (unsigned int i = 0; i < evaluations.size(); ++i) {
    for (unsigned int j = 0; j < coefficients.size(); ++j) {
      evaluations[i] +=
          coefficients[j] * std::pow(points[i], coefficients.size() - 1 - j);
    }
  }

  return evaluations;
}  
}  // namespace numeric_utils
