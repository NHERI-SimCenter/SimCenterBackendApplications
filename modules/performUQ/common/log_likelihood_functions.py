"""
Utility functions for GP_AB algorithm.

Utility functions for evaluating log-likelihoods, priors, and posterior densities
in the context of Bayesian inference using surrogate models (e.g., GP + PCA).

This module avoids nested functions to ensure compatibility with multiprocessing.
"""

import numpy as np
from scipy.stats import norm


def response_approximation(current_gp, current_pca, model_parameters):
    """
    Approximate the response using the current GP model and PCA.

    Parameters
    ----------
    current_gp : GaussianProcessModel
        The current Gaussian Process model.
    current_pca : PrincipalComponentAnalysis
        The current PCA model.
    model_parameters : np.ndarray
        The model parameters in physical space.

    Returns
    -------
    np.ndarray
        The approximated model response in the original output space.
    """
    latent_predictions, _ = current_gp.predict(model_parameters)
    return current_pca.project_back_to_original_space(latent_predictions)


def log_like(predictions, data, output_length_list):
    """
    Compute loglikelihoods for model predictions.

    Compute the log-likelihood for predictions compared to observed data,
    using weighted sum of squared errors.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions, shape (n_samples, n_outputs).
    data : np.ndarray
        Observed data, shape (n_observations, n_outputs).
    output_length_list : list of int
        Lengths of output blocks (for multi-output grouping).

    Returns
    -------
    np.ndarray
        Log-likelihoods for each sample, shape (n_samples, 1).
    """
    predictions = np.atleast_2d(predictions)
    num_rows, num_cols = data.shape
    num_samples = predictions.shape[0]

    weights = []
    start = 0
    for length in output_length_list:
        end = start + length
        y_obs = data[:, start:end]
        mse = np.mean(y_obs**2)
        weights.append(1.0 / (mse + 1e-12))
        start = end

    weighted_sse_per_sample = np.zeros(num_samples)
    start = 0
    for j, length in enumerate(output_length_list):
        end = start + length
        y_obs = data[:, start:end]
        y_obs_exp = y_obs[None, :, :]
        y_pred_exp = predictions[:, start:end][:, None, :]
        err = y_obs_exp - y_pred_exp
        sse = np.einsum('ijk,ijk->i', err, err)
        weighted_sse_per_sample += weights[j] * sse
        start = end

    exponent = -0.5 * num_rows * num_cols
    log_likes = exponent * np.log(weighted_sse_per_sample + 1e-12)
    return log_likes.reshape((num_samples, 1))


# def log_likelihood(model_parameters, log_like_fn, response_approx_fn):
#     """
#     Compute log-likelihoods for model parameters using surrogate predictions.

#     Parameters
#     ----------
#     model_parameters : np.ndarray
#         Input parameters in model space, shape (n_samples, d).
#     log_like_fn : callable
#         Function that computes log-likelihoods from predictions.
#     response_approx_fn : callable
#         Surrogate function that maps parameters to model outputs.

#     Returns
#     -------
#     np.ndarray
#         Log-likelihood values, shape (n_samples, 1).
#     """
#     predictions = response_approx_fn(model_parameters)
#     return log_like_fn(predictions)


def log_likelihood_approx(
    model_parameters, log_like_fn, response_approx_fn, simulation_number=0
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
    predictions = response_approx_fn(model_parameters)
    return log_like_fn(predictions)


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


def log_prior(model_parameters, prior_pdf_function):
    """
    Compute log-prior values from a prior PDF.

    Parameters
    ----------
    model_parameters : np.ndarray
        Model parameters, shape (n_samples, d).
    prior_pdf_function : callable
        Function returning prior PDF values.

    Returns
    -------
    np.ndarray
        Log-prior values, shape (n_samples, 1).
    """
    return np.log(prior_pdf_function(model_parameters)).reshape((-1, 1))


def log_posterior(model_parameters, log_likelihood_fn, log_prior_fn):
    """
    Compute log-posterior values by combining log-likelihood and log-prior.

    Parameters
    ----------
    model_parameters : np.ndarray
        Model parameters, shape (n_samples, d).
    log_likelihood_fn : callable
        Function computing log-likelihoods.
    log_prior_fn : callable
        Function computing log-priors.

    Returns
    -------
    np.ndarray
        Log-posterior values, shape (n_samples, 1).
    """
    ll = log_likelihood_fn(model_parameters)
    lp = log_prior_fn(model_parameters)
    return ll + lp


def log_likelihood_normal(prediction_error_vector, prediction_error_variance):
    """
    Compute log-likelihood under Gaussian assumption for prediction errors.

    Parameters
    ----------
    prediction_error_vector : np.ndarray
        Prediction errors.
    prediction_error_variance : float
        Known variance of the errors.

    Returns
    -------
    float
        Total log-likelihood value.
    """
    return np.sum(
        norm.logpdf(prediction_error_vector, 0, np.sqrt(prediction_error_variance))
    )
