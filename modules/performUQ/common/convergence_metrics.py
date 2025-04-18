"""
Functions to calculate various convergence metrics.

These include generalized KL divergence (GKL), generalized MAP (GMAP), and
Leave-One-Out Normalized Root Mean Square Error (LOO NRMSE).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# def _calculate_normalization_constants(
#     current_function_values, previous_function_values
# ):
#     """
#     Calculate normalization constants for the current and previous function values.

#     Args:
#         current_function_values (np.ndarray): The current function values.
#         previous_function_values (np.ndarray): The previous function values.

#     Returns
#     -------
#         tuple: A tuple containing the optimized alpha_1 and alpha_2 values.
#     """

#     def objective_function(log_alphas):
#         """
#         Objective function to minimize the difference from 1 for c1 and c2.

#         Args:
#             log_alphas (list): List containing log(alpha_1) and log(alpha_2).

#         Returns
#         -------
#             float: The value of the objective function.
#         """
#         log_alpha_1, log_alpha_2 = log_alphas

#         numerator_current = current_function_values - log_alpha_1
#         numerator_previous = previous_function_values - log_alpha_2

#         log_normalized_proposal = np.log(0.5) + np.logaddexp(
#             numerator_current, numerator_previous
#         )

#         c1 = (
#             1
#             / len(numerator_current)
#             * np.exp(logsumexp(numerator_current - log_normalized_proposal))
#         )
#         c2 = (
#             1
#             / len(numerator_previous)
#             * np.exp(logsumexp(numerator_previous - log_normalized_proposal))
#         )

#         return (c1 - 1) ** 2 + (c2 - 1) ** 2

#     # Initial guesses for log(alpha_1) and log(alpha_2)
#     initial_log_alpha_1 = logsumexp(current_function_values)
#     initial_log_alpha_2 = logsumexp(previous_function_values)
#     initial_guess = [initial_log_alpha_1, initial_log_alpha_2]

#     # Perform the optimization
#     result = minimize(objective_function, initial_guess, method='BFGS')

#     # Extract optimized alpha_1 and alpha_2
#     optimized_log_alpha_1, optimized_log_alpha_2 = result.x
#     alpha_1_optimized = np.exp(optimized_log_alpha_1)
#     alpha_2_optimized = np.exp(optimized_log_alpha_2)

#     return (alpha_1_optimized, alpha_2_optimized)


# def _calculate_kl_divergence(
#     current_log_likelihood_function,
#     previous_log_likelihood_function,
#     prior_log_pdf,
#     samples,
# ):
#     """
#     Calculate the KL divergence between the current and previous log target functions.

#     Args:
#         current_log_target_function (callable): The current log target function.
#         previous_log_target_function (callable): The previous log target function.
#         samples (np.ndarray): The samples to evaluate the functions.

#     Returns
#     -------
#         float: The KL divergence estimate.
#     """
#     current_log_likelihood_values = current_log_likelihood_function(samples)
#     previous_log_likelihood_values = previous_log_likelihood_function(samples)
#     current_log_posterior_density_values = (
#         current_log_likelihood_values + prior_log_pdf(samples)
#     )
#     previous_log_posterior_density_values = (
#         previous_log_likelihood_values + prior_log_pdf(samples)
#     )
#     alpha_1, alpha_2 = _calculate_normalization_constants(
#         current_log_posterior_density_values, previous_log_posterior_density_values
#     )

#     current_posterior_density_values = np.exp(current_log_posterior_density_values)
#     previous_posterior_density_values = np.exp(previous_log_posterior_density_values)
#     kl_divergence_estimate = (
#         1
#         / len(samples)
#         * np.sum(
#             np.log(
#                 current_posterior_density_values
#                 / previous_posterior_density_values
#                 * alpha_2
#                 / alpha_1
#             )
#             * current_posterior_density_values
#             / (
#                 1
#                 / 2
#                 * (
#                     current_posterior_density_values
#                     + alpha_1 / alpha_2 * previous_posterior_density_values
#                 )
#             )
#         )
#     )

#     return kl_divergence_estimate


# def calculate_gkl(
#     current_log_likelihood_function,
#     previous_log_likelihood_function,
#     prior_log_pdf,
#     samples,
# ):
#     """
#     Calculate the generalized KL divergence (GKL).

#     Args:
#         current_log_target_function (callable): The current log target function.
#         previous_log_target_function (callable): The previous log target function.
#         samples (np.ndarray): The samples to evaluate the functions.

#     Returns
#     -------
#         float: The GKL value.
#     """
#     kl_divergence_estimate = _calculate_kl_divergence(
#         current_log_likelihood_function,
#         previous_log_likelihood_function,
#         prior_log_pdf,
#         samples,
#     )
#     n_theta = np.shape(samples)[1]
#     gkl = 1 / n_theta * kl_divergence_estimate
#     return gkl


def _calculate_kl_divergence_log_alpha(
    current_log_likelihood_function,
    previous_log_likelihood_function,
    prior_log_pdf,
    samples,
):
    """KL divergence estimate using log alpha (log normalization constants)."""
    # Log posterior values
    current_log_posterior = current_log_likelihood_function(samples) + prior_log_pdf(
        samples
    )
    previous_log_posterior = previous_log_likelihood_function(
        samples
    ) + prior_log_pdf(samples)

    # Get log_alpha_1 and log_alpha_2 (stay in log-space)
    def _log_alphas():
        def obj(log_alphas):
            log_a1, log_a2 = log_alphas
            log_num_current = current_log_posterior - log_a1
            log_num_previous = previous_log_posterior - log_a2
            log_mix = np.log(0.5) + np.logaddexp(log_num_current, log_num_previous)
            c1 = np.exp(logsumexp(log_num_current - log_mix) - np.log(len(samples)))
            c2 = np.exp(logsumexp(log_num_previous - log_mix) - np.log(len(samples)))
            return (c1 - 1) ** 2 + (c2 - 1) ** 2

        init = [logsumexp(current_log_posterior), logsumexp(previous_log_posterior)]
        result = minimize(obj, init, method='BFGS')
        return result.x  # log_alpha_1, log_alpha_2

    log_alpha_1, log_alpha_2 = _log_alphas()

    # Compute the KL estimate in log space
    log_weights = current_log_posterior  # log of numerator (unnormalized)
    log_fraction = (
        current_log_posterior - previous_log_posterior + log_alpha_2 - log_alpha_1
    )

    log_mix_denom = np.log(0.5) + np.logaddexp(
        current_log_posterior, previous_log_posterior + log_alpha_1 - log_alpha_2
    )

    kl_terms = log_fraction - log_mix_denom
    kl_divergence_estimate = np.mean(np.exp(log_weights - log_mix_denom) * kl_terms)
    return kl_divergence_estimate  # noqa: RET504


def calculate_gkl(
    current_log_likelihood_function,
    previous_log_likelihood_function,
    prior_log_pdf,
    samples,
):
    """
    Compute the Generalized Kullback-Leibler (GKL) divergence between two successive
    unnormalized posterior densities using log-space calculations for numerical stability.

    The function estimates the KL divergence between the current and previous posterior
    distributions (defined as the sum of log-likelihood and log-prior) based on a shared
    set of samples, and then normalizes the divergence by the parameter dimension.

    Parameters
    ----------
    current_log_likelihood_function : callable
        Function that takes samples and returns the log-likelihood values under the current model.
    previous_log_likelihood_function : callable
        Function that takes samples and returns the log-likelihood values under the previous model.
    prior_log_pdf : callable
        Function that takes samples and returns the log-prior density values.
    samples : np.ndarray of shape (n_samples, n_parameters)
        Sample points at which the posterior log densities are evaluated.

    Returns
    -------
    float
        The GKL divergence estimate, normalized by the number of parameters.
    """  # noqa: D205
    kl_divergence_estimate = _calculate_kl_divergence_log_alpha(
        current_log_likelihood_function,
        previous_log_likelihood_function,
        prior_log_pdf,
        samples,
    )
    n_theta = samples.shape[1]
    gkl = kl_divergence_estimate / n_theta
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


# def calculate_gcv(
#     loo_predictions: np.ndarray,
#     outputs: np.ndarray,
#     weights: np.ndarray = None,
# ) -> float:
#     """
#     Calculate the weighted Leave-One-Out Normalized Root Mean Square Error (LOO NRMSE)
#     across all outputs and training points.

#     Args:
#         loo_predictions (np.ndarray): LOO predictions of shape (n_points, n_outputs).
#         true_outputs (np.ndarray): Actual outputs of shape (n_points, n_outputs).
#         weights (np.ndarray, optional): Weights for each training point. Shape: (n_points,).
#                                         If None, equal weights are used.

#     Returns
#     -------
#         float: Weighted average normalized RMS error (scalar).
#     """
#     if weights is None:
#         weights = np.ones(loo_predictions.shape[0])

#     weights = np.asarray(weights)
#     assert loo_predictions.shape == outputs.shape
#     assert weights.shape[0] == loo_predictions.shape[0]

#     # Compute normalized RMS error per output and sample
#     diff_norms = np.linalg.norm(loo_predictions - outputs, axis=1)
#     true_norms = np.linalg.norm(outputs, axis=1)

#     # Avoid division by zero
#     with np.errstate(divide='ignore', invalid='ignore'):
#         g_i = np.where(true_norms != 0, diff_norms / true_norms, 0.0)

#     # Weighted average
#     g_cv = np.sum(weights * g_i) / np.sum(weights)
#     return g_cv


def compute_standardized_rms_error(
    loo_predictions,  # shape: (n_points, total_output_dim)
    outputs,  # shape: (n_points, total_output_dim)
    output_length_list,  # list of lengths of each output group, sum = total_output_dim
):
    """
    Compute normalized RMS error for each output quantity.

    Returns
    -------
        g_i: ndarray of shape (n_points, n_outputs)
    """
    n_points = outputs.shape[0]
    n_outputs = len(output_length_list)

    g_i_matrix = np.zeros((n_points, n_outputs))
    start = 0
    for j, length in enumerate(output_length_list):
        end = start + length

        z_true = outputs[:, start:end]  # shape: (n_points, length)
        z_pred = loo_predictions[:, start:end]  # shape: (n_points, length)

        num = np.linalg.norm(z_pred - z_true, axis=1)  # shape: (n_points,)
        denom = np.linalg.norm(z_true, axis=1) + 1e-12  # avoid div-by-zero

        g_i_matrix[:, j] = num / denom
        start = end

    return g_i_matrix


def calculate_gcv(
    loo_predictions,  # shape: (n_points, total_output_dim)
    outputs,  # shape: (n_points, total_output_dim)
    output_length_list,  # list of ints, per-output dimensions
    weights=None,  # shape: (n_points,)
    weight_combination=(2 / 3, 1 / 3),
):
    """
    Compute the weighted average standardized RMS error across outputs.

    Returns
    -------
        scalar: weighted CV error g_CV
    """
    n_points = outputs.shape[0]
    # Combine weights
    weights_uniform = np.ones(n_points)
    if weights is None:
        weights = np.ones(n_points)
    alpha, beta = weight_combination
    # Safety check: weights must sum to 1
    total = alpha + beta
    if not np.isclose(total, 1.0):
        alpha /= total
        beta = 1.0 - alpha
    weights_combined = alpha * weights_uniform + beta * weights
    weights_combined /= np.sum(weights_combined)  # normalize

    # Compute g_i^(k) for each point and output
    g_i = compute_standardized_rms_error(
        loo_predictions, outputs, output_length_list
    )

    # Mean over outputs
    g_mean_per_point = np.mean(g_i, axis=1)  # shape: (n_points,)

    # Final weighted average
    g_cv = np.sum(weights_combined * g_mean_per_point)
    return g_cv
