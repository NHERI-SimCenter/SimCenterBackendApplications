import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _calculate_normalization_constants(
    current_function_values, previous_function_values
):
    # Define the objective function to minimize the difference from 1 for c1 and c2
    def objective_function(log_alphas):
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
    result = minimize(objective_function, initial_guess, method="BFGS")

    # Extract optimized alpha_1 and alpha_2
    optimized_log_alpha_1, optimized_log_alpha_2 = result.x
    alpha_1_optimized = np.exp(optimized_log_alpha_1)
    alpha_2_optimized = np.exp(optimized_log_alpha_2)

    return (alpha_1_optimized, alpha_2_optimized)


def _calculate_kl_divergence(
    current_log_target_function, previous_log_target_function, samples
):
    current_function_values = current_log_target_function(samples)
    previous_function_values = previous_log_target_function(samples)
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

    return kl_divergence_estimate


def calculate_gkl(
    current_log_target_function, previous_log_target_function, samples
):
    kl_divergence_estimate = _calculate_kl_divergence(
        current_log_target_function, previous_log_target_function, samples
    )
    n_theta = np.shape(samples)[1]
    gkl = 1 / n_theta * kl_divergence_estimate
    return gkl


def calculate_gmap(
    current_log_target_function,
    previous_log_target_function,
    samples,
    prior_variances,
):
    current_function_values = current_log_target_function(samples)
    previous_function_values = previous_log_target_function(samples)
    current_map = np.argmax(current_function_values)
    previous_map = np.argmax(previous_function_values)

    gmap = np.sqrt(np.sum((current_map - previous_map) ** 2 / prior_variances))
    return gmap


def calculate_loo_nrmse_w(
    loo_predictions,
    gp_surrogate_model_prediction,
    weights=None,
):
    if weights is None:
        weights = np.ones_like(loo_predictions)
    normalized_weights = weights / np.sum(weights)
    g_i = np.linalg.norm(
        loo_predictions - gp_surrogate_model_prediction, axis=1, keepdims=True
    ) / np.linalg.norm(gp_surrogate_model_prediction, axis=1, keepdims=True)
    g_cv = normalized_weights * g_i / len(loo_predictions)
    return g_cv
