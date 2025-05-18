"""
Utility functions for performing TMCMC (Transitional Markov Chain Monte Carlo) tests.

It includes functions for log-likelihood approximation, log-prior computation, and sample transformation.
"""

from functools import partial

import numpy as np


def log_likelihood_true(
    model_parameters, log_like_fn, response_true_fn, simulation_number=0
):
    """
    Compute log-likelihoods for model parameters using surrogate predictions.

    Parameters
    ----------
    model_parameters : np.ndarray
        Input parameters in model space, shape (n_samples, d).
    log_like_fn : callable
        Function that computes log-likelihoods from predictions.
    response_approx_fn : callable
        Surrogate function that maps parameters to model outputs.

    Returns
    -------
    np.ndarray
        Log-likelihood values, shape (n_samples, 1).
    """
    predictions = response_true_fn(simulation_number, model_parameters)
    return log_like_fn(predictions)


def response_function(simulation_number, model_parameters):
    """
    Return the input parameters as output.

    This is a placeholder and should be replaced with an actual model response function.
    """
    return model_parameters


def _ll(predictions):
    return -0.5 * np.sum((predictions - 5) ** 2, axis=1)


# Define log-likelihood function (2D Gaussian)
def _log_likelihood_function(model_parameters, simulation_number=0):
    # Assuming a Gaussian log-likelihood with mean (0, 0) and covariance identity
    return _ll(model_parameters)


_ll_function = partial(
    log_likelihood_true,
    log_like_fn=_ll,
    response_true_fn=response_function,
)


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
