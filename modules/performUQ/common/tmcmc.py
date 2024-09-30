"""Implementation of the Transitional Markov Chain Monte Carlo (TMCMC) algorithm."""

import math

import numpy as np
from scipy.special import logsumexp


def _calculate_weights_warm_start(
    beta, current_loglikelihoods, previous_loglikelihoods
):
    """
    Calculate the weights for the warm start stage.

    Args:
        beta (float): The current beta value.
        current_loglikelihoods (np.ndarray): The current log-likelihood values.
        previous_loglikelihoods (np.ndarray): The previous log-likelihood values.

    Returns
    -------
        np.ndarray: The calculated weights.
    """
    log_weights = beta * (current_loglikelihoods - previous_loglikelihoods)
    log_sum_weights = logsumexp(log_weights)
    normalized_log_weights = log_weights - log_sum_weights
    normalized_weights = np.exp(normalized_log_weights)
    weights = normalized_weights / np.sum(normalized_weights)
    return weights  # noqa: RET504


def calculate_warm_start_stage(
    current_loglikelihood_approximation, previous_results, threshold_cov=1
):
    """
    Calculate the warm start stage number based on the coefficient of variation of weights.

    Args:
        current_loglikelihood_approximation (callable): Function to approximate current log-likelihoods.
        previous_results (tuple): The previous results containing samples, betas, and log-likelihoods.
        threshold_cov (float, optional): The threshold for the coefficient of variation. Defaults to 1.

    Returns
    -------
        int: The stage number for the warm start.
    """
    stage_nums = sorted(previous_results[0].keys(), reverse=True)
    for stage_num in stage_nums:
        current_loglikelihoods = current_loglikelihood_approximation(
            previous_results[0][stage_num]
        )
        previous_loglikelihoods = previous_results[2][stage_num]
        beta = previous_results[1][stage_num]
        weights = _calculate_weights_warm_start(
            beta, current_loglikelihoods, previous_loglikelihoods
        )
        cov_weights = np.nanstd(weights) / np.nanmean(weights)
        if cov_weights < threshold_cov:
            return stage_num
    return 0


def _calculate_weights(beta_increment, log_likelihoods):
    """
    Calculate the weights for the given beta increment and log-likelihoods.

    Args:
        beta_increment (float): The increment in beta.
        log_likelihoods (np.ndarray): The log-likelihood values.

    Returns
    -------
        np.ndarray: The calculated weights.
    """
    log_weights = beta_increment * log_likelihoods
    log_sum_weights = logsumexp(log_weights)
    normalized_log_weights = log_weights - log_sum_weights
    normalized_weights = np.exp(normalized_log_weights)
    weights = normalized_weights / np.sum(normalized_weights)
    return weights  # noqa: RET504


def _calculate_log_evidence(beta_increment, log_likelihoods):
    """
    Calculate the log evidence for the given beta increment and log-likelihoods.

    Args:
        beta_increment (float): The increment in beta.
        log_likelihoods (np.ndarray): The log-likelihood values.

    Returns
    -------
        float: The calculated log evidence.
    """
    log_evidence = logsumexp(beta_increment * log_likelihoods) - np.log(
        len(log_likelihoods)
    )
    return log_evidence  # noqa: RET504


def _increment_beta(log_likelihoods, beta, threshold_cov=1):
    """
    Increment the beta value based on the coefficient of variation of weights.

    Args:
        log_likelihoods (np.ndarray): The log-likelihood values.
        beta (float): The current beta value.
        threshold_cov (float, optional): The threshold for the coefficient of variation. Defaults to 1.

    Returns
    -------
        float: The new beta value.
    """
    if beta >= 1:
        return 1
    beta_increment = 1 - beta
    weights = _calculate_weights(beta_increment, log_likelihoods)
    cov_weights = np.nanstd(weights) / np.nanmean(weights)
    while cov_weights > threshold_cov:
        beta_increment = 0.99 * beta_increment
        weights = _calculate_weights(beta_increment, log_likelihoods)
        cov_weights = np.nanstd(weights) / np.nanmean(weights)
    proposed_beta = beta + beta_increment
    new_beta = min(proposed_beta, 1)
    return new_beta  # noqa: RET504


def _get_scaled_proposal_covariance(samples, weights, scale_factor=0.2):
    """
    Calculate the scaled proposal covariance matrix.

    Args:
        samples (np.ndarray): The samples.
        weights (np.ndarray): The weights.
        scale_factor (float, optional): The scale factor. Defaults to 0.2.

    Returns
    -------
        np.ndarray: The scaled proposal covariance matrix.
    """
    return scale_factor**2 * np.cov(
        samples, rowvar=False, aweights=weights.flatten()
    )


class TMCMC:
    """
    A class to perform Transitional Markov Chain Monte Carlo (TMCMC) sampling.

    Attributes
    ----------
        _log_likelihood_approximation (callable): Function to approximate log-likelihoods.
        _log_posterior_approximation (callable): Function to approximate log-posterior densities.
        num_steps (int): The number of steps for each stage.
        threshold_cov (float): The threshold for the coefficient of variation.
        thinning_factor (int): The thinning factor for the MCMC chain.
        adapt_frequency (int): The frequency of adaptation for the proposal distribution.
    """

    def __init__(
        self,
        log_likelihood_approximation_function,
        log_target_density_approximation_function,
        threshold_cov=1,
        num_steps=1,
        thinning_factor=10,
        adapt_frequency=100,
    ):
        """
        Initialize the TMCMC class.

        Args:
            log_likelihood_approximation_function (callable): Function to approximate log-likelihoods.
            log_target_density_approximation_function (callable): Function to approximate log-posterior densities.
            threshold_cov (float, optional): The threshold for the coefficient of variation. Defaults to 1.
            num_steps (int, optional): The number of steps for each stage. Defaults to 1.
            thinning_factor (int, optional): The thinning factor for the MCMC chain. Defaults to 10.
            adapt_frequency (int, optional): The frequency of adaptation for the proposal distribution. Defaults to 100.
        """
        self._log_likelihood_approximation = log_likelihood_approximation_function
        self._log_posterior_approximation = log_target_density_approximation_function

        self.num_steps = num_steps
        self.threshold_cov = threshold_cov
        self.thinning_factor = thinning_factor
        self.adapt_frequency = adapt_frequency

    def _run_one_stage(
        self,
        samples,
        log_likelihoods,
        log_target_density_values,
        beta,
        rng,
        log_likelihood_function,
        log_target_density_function,
        scale_factor,
        target_acceptance_rate,
        do_thinning=False,  # noqa: FBT002
        burn_in_steps=0,
    ):
        """
        Run one stage of the TMCMC algorithm.

        Args:
            samples (np.ndarray): The samples.
            log_likelihoods (np.ndarray): The log-likelihood values.
            log_target_density_values (np.ndarray): The log-target density values.
            beta (float): The current beta value.
            rng (np.random.Generator): The random number generator.
            log_likelihood_function (callable): Function to calculate log-likelihoods.
            log_target_density_function (callable): Function to calculate log-target densities.
            scale_factor (float): The scale factor for the proposal distribution.
            target_acceptance_rate (float): The target acceptance rate for the MCMC chain.
            do_thinning (bool, optional): Whether to perform thinning. Defaults to False.
            burn_in_steps (int, optional): The number of burn-in steps. Defaults to 0.

        Returns
        -------
            tuple: A tuple containing the new samples, new log-likelihoods, new log-target density values, new beta, and log evidence.
        """
        new_beta = _increment_beta(log_likelihoods, beta)
        log_evidence = _calculate_log_evidence(new_beta - beta, log_likelihoods)
        weights = _calculate_weights(new_beta - beta, log_likelihoods)

        proposal_covariance = _get_scaled_proposal_covariance(
            samples, weights, scale_factor
        )

        new_samples = np.zeros_like(samples)
        new_log_likelihoods = np.zeros_like(log_likelihoods)
        new_log_target_density_values = np.zeros_like(log_target_density_values)

        current_samples = samples.copy()
        current_log_likelihoods = log_likelihoods.copy()
        current_log_target_density_values = log_target_density_values.copy()

        num_samples = samples.shape[0]
        num_accepts = 0
        n_adapt = 1
        step_count = 0
        for k in range(burn_in_steps + num_samples):
            # print(f'{k=}')
            index = rng.choice(num_samples, p=weights.flatten())
            if k >= burn_in_steps:
                if new_beta == 1 or do_thinning:
                    self.num_steps = self.num_steps * self.thinning_factor
            for _ in range(self.num_steps):
                step_count += 1
                if step_count % self.adapt_frequency == 0:
                    acceptance_rate = num_accepts / self.adapt_frequency
                    num_accepts = 0
                    n_adapt += 1
                    ca = (acceptance_rate - target_acceptance_rate) / (
                        math.sqrt(n_adapt)
                    )
                    scale_factor = scale_factor * np.exp(ca)
                    proposal_covariance = _get_scaled_proposal_covariance(
                        current_samples, weights, scale_factor
                    )

                proposed_state = rng.multivariate_normal(
                    current_samples[index, :], proposal_covariance
                ).reshape(1, -1)
                log_likelihood_at_proposed_state = log_likelihood_function(
                    proposed_state
                )
                log_target_density_at_proposed_state = log_target_density_function(
                    proposed_state, log_likelihood_at_proposed_state
                )
                log_hastings_ratio = (
                    log_target_density_at_proposed_state
                    - current_log_target_density_values[index]
                )
                u = rng.uniform()
                accept = np.log(u) <= log_hastings_ratio
                if accept:
                    num_accepts += 1
                    current_samples[index, :] = proposed_state
                    current_log_likelihoods[index] = log_likelihood_at_proposed_state
                    current_log_target_density_values[index] = (
                        log_target_density_at_proposed_state
                    )
                    if k >= burn_in_steps:
                        weights = _calculate_weights(
                            new_beta - beta, current_log_likelihoods
                        )
            if k >= burn_in_steps:
                k_prime = k - burn_in_steps
                new_samples[k_prime, :] = current_samples[index, :]
                new_log_likelihoods[k_prime] = current_log_likelihoods[index]
                new_log_target_density_values[k_prime] = (
                    current_log_target_density_values[index]
                )

        return (
            new_samples,
            new_log_likelihoods,
            new_log_target_density_values,
            new_beta,
            log_evidence,
        )

    def run(
        self,
        samples_dict,
        betas_dict,
        log_likelihoods_dict,
        log_target_density_values_dict,
        log_evidence_dict,
        rng,
        stage_num,
        num_burn_in=0,
    ):
        """
        Run the TMCMC algorithm.

        Args:
            samples_dict (dict): Dictionary of samples for each stage.
            betas_dict (dict): Dictionary of beta values for each stage.
            log_likelihoods_dict (dict): Dictionary of log-likelihood values for each stage.
            log_target_density_values_dict (dict): Dictionary of log-target density values for each stage.
            log_evidence_dict (dict): Dictionary of log evidence values for each stage.
            rng (np.random.Generator): The random number generator.
            stage_num (int): The current stage number.
            num_burn_in (int, optional): The number of burn-in steps. Defaults to 0.

        Returns
        -------
            tuple: A tuple containing the updated dictionaries for samples, betas, log-likelihoods, log-target density values, and log evidence.
        """
        self.num_dimensions = samples_dict[0].shape[1]
        self.target_acceptance_rate = 0.23 + 0.21 / self.num_dimensions
        self.scale_factor = 2.4 / np.sqrt(self.num_dimensions)
        while betas_dict[stage_num] < 1:
            # print(f'Stage {stage_num}')
            (
                new_samples,
                new_log_likelihoods,
                new_log_target_density_values,
                new_beta,
                log_evidence,
            ) = self._run_one_stage(
                samples_dict[stage_num],
                log_likelihoods_dict[stage_num],
                log_target_density_values_dict[stage_num],
                betas_dict[stage_num],
                rng,
                self._log_likelihood_approximation,
                self._log_posterior_approximation,
                self.scale_factor,
                self.target_acceptance_rate,
                do_thinning=False,
                burn_in_steps=num_burn_in,
            )
            stage_num += 1
            samples_dict[stage_num] = new_samples
            betas_dict[stage_num] = new_beta
            log_likelihoods_dict[stage_num] = new_log_likelihoods
            log_target_density_values_dict[stage_num] = new_log_target_density_values
            log_evidence_dict[stage_num] = log_evidence

        return (
            samples_dict,
            betas_dict,
            log_likelihoods_dict,
            log_target_density_values_dict,
            log_evidence_dict,
        )
