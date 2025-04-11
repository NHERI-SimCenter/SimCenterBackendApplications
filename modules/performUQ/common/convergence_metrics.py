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
    current_log_target_density_values = (
        current_log_likelihood_values + prior_log_pdf(samples)
    )
    previous_log_target_density_values = (
        previous_log_likelihood_values + prior_log_pdf(samples)
    )
    alpha_1, alpha_2 = _calculate_normalization_constants(
        current_log_target_density_values, previous_log_target_density_values
    )

    current_target_density_values = np.exp(current_log_target_density_values)
    previous_target_density_values = np.exp(previous_log_target_density_values)
    kl_divergence_estimate = (
        1
        / len(samples)
        * np.sum(
            np.log(
                current_target_density_values
                / previous_target_density_values
                * alpha_2
                / alpha_1
            )
            * current_target_density_values
            / (
                1
                / 2
                * (
                    current_target_density_values
                    + alpha_1 / alpha_2 * previous_target_density_values
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
    Calculate the normalized RMSE between MAP estimates (g_MAP) across two iterations.

    Args:
        current_log_likelihood_function (callable): Log-likelihood function for current iteration.
        previous_log_likelihood_function (callable): Log-likelihood function for previous iteration.
        prior_log_pdf (callable): Function computing log prior density.
        samples (np.ndarray): Array of shape (n_samples, n_params), samples to evaluate.
        prior_variances (np.ndarray): Array of shape (n_params,), prior variances for normalization.

    Returns
    -------
        float: g_MAP value.
    """
    # Evaluate posterior log-probabilities (unnormalized)
    current_log_post = current_log_likelihood_function(samples).reshape(
        -1
    ) + prior_log_pdf(samples).reshape(-1)
    previous_log_post = previous_log_likelihood_function(samples).reshape(
        -1
    ) + prior_log_pdf(samples).reshape(-1)

    # MAP estimates: sample with max log-posterior
    current_map = samples[np.argmax(current_log_post)]
    previous_map = samples[np.argmax(previous_log_post)]

    # Compute normalized squared differences
    delta = current_map - previous_map
    normalized_squared_diff = delta**2 / prior_variances

    # g_MAP: Root mean of normalized squared differences
    gmap = np.sqrt(np.mean(normalized_squared_diff))

    return gmap  # noqa: RET504


def calculate_gcv(
    loo_predictions: np.ndarray,
    outputs: np.ndarray,
    weights: np.ndarray = None,
) -> float:
    """
    Calculate the weighted Leave-One-Out Normalized Root Mean Square Error (LOO NRMSE)
    across all outputs and training points.

    Args:
        loo_predictions (np.ndarray): LOO predictions of shape (n_points, n_outputs).
        true_outputs (np.ndarray): Actual outputs of shape (n_points, n_outputs).
        weights (np.ndarray, optional): Weights for each training point. Shape: (n_points,).
                                        If None, equal weights are used.

    Returns
    -------
        float: Weighted average normalized RMS error (scalar).
    """  # noqa: D205
    if weights is None:
        weights = np.ones(loo_predictions.shape[0])

    weights = np.asarray(weights)
    assert loo_predictions.shape == outputs.shape
    assert weights.shape[0] == loo_predictions.shape[0]

    # Compute normalized RMS error per output and sample
    diff_norms = np.linalg.norm(loo_predictions - outputs, axis=1)
    true_norms = np.linalg.norm(outputs, axis=1)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        g_i = np.where(true_norms != 0, diff_norms / true_norms, 0.0)

    # Weighted average
    g_cv = np.sum(weights * g_i) / np.sum(weights)
    return g_cv  # noqa: RET504
