"""
Functions to calculate various convergence metrics.

These include generalized KL divergence (GKL), generalized MAP (GMAP), and
Leave-One-Out Normalized Root Mean Square Error (LOO NRMSE).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _calculate_normalization_constants(
    current_function_values, previous_function_values
):
    """
    Calculate normalization constants for the current and previous function values.

    Args:
        current_function_values (np.ndarray): The current function values.
        previous_function_values (np.ndarray): The previous function values.

    Returns
    -------
        tuple: A tuple containing the optimized alpha_1 and alpha_2 values.
    """

    def objective_function(log_alphas):
        """
        Objective function to minimize the difference from 1 for c1 and c2.

        Args:
            log_alphas (list): List containing log(alpha_1) and log(alpha_2).

        Returns
        -------
            float: The value of the objective function.
        """
        log_alpha_1, log_alpha_2 = log_alphas

        numerator_current = current_function_values - log_alpha_1
        numerator_previous = previous_function_values - log_alpha_2

        log_normalized_proposal = np.log(0.5) + np.logaddexp(
            numerator_current, numerator_previous
        )

        c1 = (
            1
            / len(numerator_current)
            * np.exp(logsumexp(numerator_current - log_normalized_proposal))
        )
        c2 = (
            1
            / len(numerator_previous)
            * np.exp(logsumexp(numerator_previous - log_normalized_proposal))
        )

        return (c1 - 1) ** 2 + (c2 - 1) ** 2

    # Initial guesses for log(alpha_1) and log(alpha_2)
    initial_log_alpha_1 = logsumexp(current_function_values)
    initial_log_alpha_2 = logsumexp(previous_function_values)
    initial_guess = [initial_log_alpha_1, initial_log_alpha_2]

    # Perform the optimization
    result = minimize(objective_function, initial_guess, method='BFGS')

    # Extract optimized alpha_1 and alpha_2
    optimized_log_alpha_1, optimized_log_alpha_2 = result.x
    alpha_1_optimized = np.exp(optimized_log_alpha_1)
    alpha_2_optimized = np.exp(optimized_log_alpha_2)

    return (alpha_1_optimized, alpha_2_optimized)


def _calculate_kl_divergence(
    current_log_likelihood_function,
    previous_log_likelihood_function,
    prior_log_pdf,
    samples,
):
    """
    Calculate the KL divergence between the current and previous log target functions.

    Args:
        current_log_target_function (callable): The current log target function.
        previous_log_target_function (callable): The previous log target function.
        samples (np.ndarray): The samples to evaluate the functions.

    Returns
    -------
        float: The KL divergence estimate.
    """
    current_log_likelihood_values = current_log_likelihood_function(samples)
    previous_log_likelihood_values = previous_log_likelihood_function(samples)
    current_function_values = current_log_likelihood_values + prior_log_pdf(samples)
    previous_function_values = previous_log_likelihood_values + prior_log_pdf(
        samples
    )

    alpha_1, alpha_2 = _calculate_normalization_constants(
        current_function_values, previous_function_values
    )
    kl_divergence_estimate = (
        1
        / len(samples)
        * np.sum(
            np.log(
                current_function_values
                / previous_function_values
                * alpha_2
                / alpha_1
            )
            * current_function_values
            / (
                1
                / 2
                * (
                    current_function_values
                    + alpha_1 / alpha_2 * previous_function_values
                )
            )
        )
    )

    return kl_divergence_estimate  # noqa: RET504


def calculate_gkl(
    current_log_likelihood_function,
    previous_log_likelihood_function,
    prior_log_pdf,
    samples,
):
    """
    Calculate the generalized KL divergence (GKL).

    Args:
        current_log_target_function (callable): The current log target function.
        previous_log_target_function (callable): The previous log target function.
        samples (np.ndarray): The samples to evaluate the functions.

    Returns
    -------
        float: The GKL value.
    """
    kl_divergence_estimate = _calculate_kl_divergence(
        current_log_likelihood_function,
        previous_log_likelihood_function,
        prior_log_pdf,
        samples,
    )
    n_theta = np.shape(samples)[1]
    gkl = 1 / n_theta * kl_divergence_estimate
    return gkl  # noqa: RET504


def calculate_gmap(
    current_log_likelihood_function,
    previous_log_likelihood_function,
    prior_log_pdf,
    samples,
    prior_variances,
):
    """
    Calculate the generalized MAP (GMAP).

    Args:
        current_log_target_function (callable): The current log target function.
        previous_log_target_function (callable): The previous log target function.
        samples (np.ndarray): The samples to evaluate the functions.
        prior_variances (np.ndarray): The prior variances.

    Returns
    -------
        float: The GMAP value.
    """
    current_log_likelihood_values = current_log_likelihood_function(samples)
    previous_log_likelihood_values = previous_log_likelihood_function(samples)
    current_map = np.argmax(current_log_likelihood_values + prior_log_pdf(samples))
    previous_map = np.argmax(previous_log_likelihood_values + prior_log_pdf(samples))

    gmap = np.sqrt(np.sum((current_map - previous_map) ** 2 / prior_variances))
    return gmap  # noqa: RET504


def calculate_loo_nrmse_w(
    loo_predictions,
    gp_surrogate_model_prediction,
    weights=None,
):
    """
    Calculate the Leave-One-Out Normalized Root Mean Square Error (LOO NRMSE) with weights.

    Args:
        loo_predictions (np.ndarray): The LOO predictions.
        gp_surrogate_model_prediction (np.ndarray): The GP surrogate model predictions.
        weights (np.ndarray, optional): The weights for the predictions. Defaults to None.

    Returns
    -------
        np.ndarray: The LOO NRMSE values.
    """
    if weights is None:
        weights = np.ones_like(loo_predictions)
    normalized_weights = weights / np.sum(weights)
    g_i = np.linalg.norm(
        loo_predictions - gp_surrogate_model_prediction, axis=1, keepdims=True
    ) / np.linalg.norm(gp_surrogate_model_prediction, axis=1, keepdims=True)
    g_cv = normalized_weights * g_i / len(loo_predictions)
    return g_cv  # noqa: RET504
