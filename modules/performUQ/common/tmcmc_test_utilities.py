"""
Utility functions for performing TMCMC (Transitional Markov Chain Monte Carlo) tests.

It includes functions for log-likelihood approximation, log-prior computation, and sample transformation.
"""

import numpy as np


# Define log-likelihood function (2D Gaussian)
def _log_likelihood_approximation_function(model_parameters):
    # Assuming a Gaussian log-likelihood with mean (0, 0) and covariance identity
    return -0.5 * np.sum((model_parameters - 5) ** 2, axis=1)


# Define log-prior function (assume uniform)
def _log_prior_pdf(model_parameters):
    lower_bound = -10
    upper_bound = 10

    # Ensure it's 2D
    model_parameters = np.atleast_2d(model_parameters)

    log_priors = []
    for parameter in model_parameters:
        if not np.all((parameter >= lower_bound) & (parameter <= upper_bound)):
            log_priors.append(-np.inf)
        else:
            log_priors.append(np.log(1 / (upper_bound - lower_bound)))
    return np.array(log_priors)


# Define sample transformation function (in this case, this does nothing to the sample)
def _sample_transformation_function(samples):
    return samples
