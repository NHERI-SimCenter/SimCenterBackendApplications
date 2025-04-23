"""Implementation of the Transitional Markov Chain Monte Carlo (TMCMC) algorithm."""

import logging
import math
import time
from collections import Counter

import numpy as np
import uq_utilities
from numpy.random import SeedSequence, default_rng
from safer_cholesky import SaferCholesky
from scipy.optimize import root_scalar
from scipy.special import logsumexp

safer_cholesky = SaferCholesky(debug=True)


def _calculate_weights_warm_start(
    beta, current_loglikelihood_values, previous_loglikelihood_values
):
    """
    Calculate the weights for the warm start stage.

    Args:
        beta (float): The current beta value.
        current_loglikelihood_values (np.ndarray): The current log-likelihood values.
        previous_loglikelihood_values (np.ndarray): The previous log-likelihood values.

    Returns
    -------
        np.ndarray: The calculated weights.
    """
    log_weights = beta * (
        current_loglikelihood_values - previous_loglikelihood_values
    )
    normalized_log_weights = log_weights - np.max(log_weights)
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
    stage_nums = sorted(previous_results['samples_dict'].keys(), reverse=True)
    for stage_num in stage_nums:
        current_loglikelihood_values = current_loglikelihood_approximation(
            previous_results['model_parameters_dict'][stage_num]
        )
        previous_loglikelihood_values = previous_results[
            'log_likelihood_values_dict'
        ][stage_num]
        beta = previous_results['betas_dict'][stage_num]
        weights = _calculate_weights_warm_start(
            beta, current_loglikelihood_values, previous_loglikelihood_values
        )
        cov_weights = np.nanstd(weights) / np.nanmean(weights)
        if cov_weights < threshold_cov:
            return stage_num
    return 0


def _calculate_weights(beta_increment, log_likelihood_values):
    """
    Calculate the weights for the given beta increment and log-likelihoods.

    Args:
        beta_increment (float): The increment in beta.
        log_likelihood_values (np.ndarray): The log-likelihood values.

    Returns
    -------
        np.ndarray: The calculated weights.
    """
    log_weights = beta_increment * log_likelihood_values
    normalized_log_weights = log_weights - np.max(log_weights)
    normalized_weights = np.exp(normalized_log_weights)
    weights = normalized_weights / np.sum(normalized_weights)
    return weights  # noqa: RET504


def _calculate_log_evidence(beta_increment, log_likelihood_values):
    """
    Calculate the log evidence for the given beta increment and log-likelihoods.

    Args:
        beta_increment (float): The increment in beta.
        log_likelihood_values (np.ndarray): The log-likelihood values.

    Returns
    -------
        float: The calculated log evidence.
    """
    log_evidence = logsumexp(beta_increment * log_likelihood_values) - np.log(
        len(log_likelihood_values)
    )
    return log_evidence  # noqa: RET504


def _increment_beta(log_likelihood_values, beta, threshold_cov=1):
    """
    Attempt to increment beta using optimization. If optimization fails, fall back to trial-and-error.

    Args:
        log_likelihood_values (np.ndarray): The log-likelihood values.
        beta (float): The current beta value.
        threshold_cov (float, optional): The threshold for the coefficient of variation. Defaults to 1.

    Returns
    -------
        float: The new beta value.
    """

    def cov_objective(beta_increment):
        weights = _calculate_weights(beta_increment, log_likelihood_values)
        cov_weights = np.nanstd(weights) / np.nanmean(weights)
        return cov_weights - threshold_cov

    # print(f'{cov_objective(0) = }, {cov_objective(1 - beta) = }')
    # Check if optimization method is feasible
    if np.sign(cov_objective(0)) == np.sign(cov_objective(1 - beta)):
        # If signs are the same, set beta to its maximum possible value of 1
        # print('Optimization not feasible. Setting beta to 1.0')
        # wts = _calculate_weights(1 - beta, log_likelihoods)
        # print(f'cov_weights at new beta = {np.nanstd(wts) / np.nanmean(wts)}')
        return 1.0

    # Try optimization method first
    result = root_scalar(cov_objective, bracket=[0, 1 - beta], method='bisect')

    if result.converged:
        # If optimization succeeds, calculate the new beta
        new_beta = min(beta + result.root, 1)
        # wts = _calculate_weights(result.root, log_likelihoods)
        # print(f'cov_weights at new beta = {np.nanstd(wts) / np.nanmean(wts)}')
        return new_beta  # noqa: RET504

    # Fallback to trial-and-error approach if optimization fails
    # print('Optimization failed. Fallback to trial-and-error approach.')
    beta_increment = 1 - beta
    weights = _calculate_weights(beta_increment, log_likelihood_values)
    cov_weights = np.nanstd(weights) / np.nanmean(weights)

    while cov_weights > threshold_cov:
        beta_increment = 0.99 * beta_increment
        weights = _calculate_weights(beta_increment, log_likelihood_values)
        cov_weights = np.nanstd(weights) / np.nanmean(weights)

    proposed_beta = beta + beta_increment
    new_beta = min(proposed_beta, 1)
    return new_beta  # noqa: RET504


def get_scaled_proposal_covariance(
    samples, weights, scale_factor=0.2, min_eigval=1e-10
):
    """
    Compute a scaled proposal covariance matrix with robust fallback handling.

    This function:
    - Computes weighted covariance
    - Falls back to unweighted if needed
    - Regularizes via eigendecomposition (or SVD fallback)

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n_samples, n_dimensions).
    weights : np.ndarray
        Normalized weights of shape (n_samples,) or (n_samples, 1).
    scale_factor : float, optional
        Scaling factor for the proposal covariance. Default is 0.2.
    min_eigval : float, optional
        Minimum eigenvalue or singular value threshold. Default is 1e-10.

    Returns
    -------
    cov_scaled : np.ndarray
        Scaled, regularized proposal covariance matrix.
    """
    weights = np.asarray(weights).flatten()

    # Step 1: Attempt weighted covariance
    cov = np.cov(samples, rowvar=False, aweights=weights)
    if not np.all(np.isfinite(cov)):
        cov = np.cov(samples, rowvar=False)

    # Step 2: Ensure symmetry
    cov = 0.5 * (cov + cov.T)

    # Step 3: Try eigendecomposition with clipping
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues_clipped = np.clip(eigenvalues, min_eigval, None)
        cov_reg = np.real(
            eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
        )
    except np.linalg.LinAlgError:
        # Step 4: Fall back to SVD
        left_vectors, singular_values, _ = np.linalg.svd(cov)
        singular_values_clipped = np.clip(singular_values, min_eigval, None)
        cov_reg = np.real((left_vectors * singular_values_clipped) @ left_vectors.T)

    cov_scaled = scale_factor**2 * cov_reg
    return cov_scaled  # noqa: RET504


def _generate_error_message(size):
    """Generate an error message for the given size."""
    return f'Expected a single value, but got {size} values.'


def run_one_stage_unequal_chain_lengths(  # noqa: C901, PLR0913
    samples,
    model_parameters,
    log_likelihood_values,
    log_target_density_values,
    beta,
    rng,
    log_likelihood_function,
    log_prior_density_function,
    sample_transformation_function,
    scale_factor,
    target_acceptance_rate,
    do_thinning=False,  # noqa: FBT002
    burn_in_steps=0,
    number_of_steps=1,
    thinning_factor=1,
    adapt_frequency=50,
    logger=None,
):
    """
    Run one stage of the TMCMC algorithm.

    Args:
        samples (np.ndarray): The samples.
        log_likelihood_values (np.ndarray): The log-likelihood values.
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
    new_beta = _increment_beta(log_likelihood_values, beta)
    log_evidence = _calculate_log_evidence(new_beta - beta, log_likelihood_values)
    weights = _calculate_weights(new_beta - beta, log_likelihood_values)

    proposal_covariance = get_scaled_proposal_covariance(
        samples, weights, scale_factor
    )
    try:
        # cholesky_lower_triangular_matrix = np.linalg.cholesky(proposal_covariance)
        cholesky_lower_triangular_matrix = safer_cholesky.decompose(
            proposal_covariance
        )
    except np.linalg.LinAlgError as exc:
        msg = f'Cholesky decomposition failed: {exc}'
        raise RuntimeError(msg) from exc

    new_samples = np.zeros_like(samples)
    new_model_parameters = np.zeros_like(model_parameters)
    new_log_likelihood_values = np.zeros_like(log_likelihood_values)
    new_log_target_density_values = np.zeros_like(log_target_density_values)

    current_samples = samples.copy()
    current_model_parameters = model_parameters.copy()
    current_log_likelihood_values = log_likelihood_values.copy()
    current_log_target_density_values = log_target_density_values.copy()

    num_samples = samples.shape[0]
    num_accepts = 0
    num_adapt = 1
    step_count = 0
    num_steps = number_of_steps
    # print(f'{new_beta = }, {do_thinning = }, {num_steps = }')

    index_counter = Counter()
    for k in range(burn_in_steps + num_samples):
        index = rng.choice(num_samples, p=weights.flatten())
        index_counter[index] += 1
        if k >= burn_in_steps:
            if new_beta == 1 or do_thinning:
                num_steps = number_of_steps * thinning_factor
        # print(f'{new_beta = }, {do_thinning = }, {num_steps = }')
        for _ in range(num_steps):
            step_count += 1
            if step_count % adapt_frequency == 0:
                acceptance_rate = num_accepts / adapt_frequency
                num_accepts = 0
                num_adapt += 1
                ca = (acceptance_rate - target_acceptance_rate) / (
                    math.sqrt(num_adapt)
                )
                scale_factor = scale_factor * np.exp(ca)
                proposal_covariance = get_scaled_proposal_covariance(
                    current_samples, weights, scale_factor
                )
                try:
                    # cholesky_lower_triangular_matrix = np.linalg.cholesky(proposal_covariance)
                    cholesky_lower_triangular_matrix = safer_cholesky.decompose(
                        proposal_covariance
                    )
                except np.linalg.LinAlgError as exc:
                    msg = f'Cholesky decomposition failed: {exc}'
                    raise RuntimeError(msg) from exc

            standard_normal_samples = rng.standard_normal(
                size=current_samples.shape[1]
            )
            proposed_state = (
                current_samples[index, :]
                + cholesky_lower_triangular_matrix @ standard_normal_samples
            ).reshape(1, -1)

            proposed_model_parameter = np.reshape(
                sample_transformation_function(proposed_state), proposed_state.shape
            )
            log_likelihood_at_proposed_model_parameter = log_likelihood_function(
                proposed_model_parameter, simulation_number=step_count
            )
            log_prior_density_at_proposed_model_parameter = (
                log_prior_density_function(proposed_model_parameter)
            )
            log_target_density_at_proposed_model_parameter = (
                new_beta * log_likelihood_at_proposed_model_parameter
                + log_prior_density_at_proposed_model_parameter
            )

            current_model_parameter = current_model_parameters[index, :]
            log_likelihood_at_current_model_parameter = (
                current_log_likelihood_values[index]
            )
            log_prior_density_at_current_parameter = log_prior_density_function(
                current_model_parameter
            )
            log_target_density_at_current_parameter = (
                new_beta * log_likelihood_at_current_model_parameter
                + log_prior_density_at_current_parameter
            )

            log_hastings_ratio = (
                log_target_density_at_proposed_model_parameter
                - log_target_density_at_current_parameter
            )
            u = rng.uniform()
            accept = np.log(u) <= log_hastings_ratio
            # print('accept:', accept, type(accept), np.shape(accept))

            if accept:
                num_accepts += 1
                current_samples[index, :] = proposed_state
                current_model_parameters[index, :] = proposed_model_parameter
                # current_log_likelihoods[index] = log_likelihood_at_proposed_state
                if log_likelihood_at_proposed_model_parameter.size != 1:
                    msg = _generate_error_message(
                        log_likelihood_at_proposed_model_parameter.size
                    )
                    raise ValueError(msg)
                current_log_likelihood_values[index] = (
                    log_likelihood_at_proposed_model_parameter.item()
                )
                if log_target_density_at_proposed_model_parameter.size != 1:
                    msg = _generate_error_message(
                        log_target_density_at_proposed_model_parameter.size
                    )
                    raise ValueError(msg)
                current_log_target_density_values[index] = (
                    log_target_density_at_proposed_model_parameter.item()
                )
                if k >= burn_in_steps:
                    weights = _calculate_weights(
                        new_beta - beta, current_log_likelihood_values
                    )
        if k >= burn_in_steps:
            k_prime = k - burn_in_steps
            new_samples[k_prime, :] = current_samples[index, :]
            new_model_parameters[k_prime, :] = current_model_parameters[index, :]
            new_log_likelihood_values[k_prime] = current_log_likelihood_values[index]
            new_log_target_density_values[k_prime] = (
                current_log_target_density_values[index]
            )
    # total_num_model_evaluations = (burn_in_steps + num_samples) * num_steps
    total_num_model_evaluations = step_count
    num_unique_indices = len(index_counter)
    max_count = max(index_counter.values())
    min_count = min(index_counter.values())
    if logger is not None:
        logger.info(
            f'    > Number of chains picked = {num_unique_indices} out of {num_samples}'
        )
        logger.info(f'    > Longest chain length = {max_count}')
        logger.info(f'    > Shortest chain length = {min_count}')
        logger.info(
            f'    > Total number of model evaluations = {total_num_model_evaluations}'
        )
    return (
        new_samples,
        new_model_parameters,
        new_log_likelihood_values,
        new_log_target_density_values,
        new_beta,
        log_evidence,
        total_num_model_evaluations,
    )


def metropolis_step(
    current_x,  # shape: (1, d)
    current_x_model,  # shape: (1, d)
    loglike_current,
    logtarget_current,
    proposal_chol,
    beta,
    rng,
    log_likelihood_fn,
    log_prior_fn,
    sample_transformation_fn,
    step_num=0,
):
    """
    Perform one Metropolis-Hastings step using a Cholesky-based Gaussian proposal.

    Parameters
    ----------
    current_x : np.ndarray
        Current sample in latent space, shape (1, d).
    current_x_model : np.ndarray
        Corresponding model-space sample, shape (1, d).
    loglike_current : float
        Log-likelihood of the current model-space sample.
    logtarget_current : float
        Log of the tempered target density at the current sample.
    proposal_chol : np.ndarray
        Lower Cholesky factor of the proposal covariance matrix, shape (d, d).
    beta : float
        Current tempering parameter for TMCMC.
    rng : np.random.Generator
        Random number generator instance.
    log_likelihood_fn : callable
        Function that computes log-likelihood given a model-space sample of shape (1, d).
    log_prior_fn : callable
        Function that computes log-prior given a model-space sample of shape (1, d).
    sample_transformation_fn : callable
        Function mapping a latent-space sample of shape (1, d) to model space (1, d).

    Returns
    -------
    proposal_x : np.ndarray
        Accepted or rejected sample in latent space, shape (1, d).
    proposal_model : np.ndarray
        Corresponding sample in model space, shape (1, d).
    loglike : float
        Log-likelihood of the accepted sample.
    logtarget : float
        Log of the tempered target density of the accepted sample.
    """
    if current_x.ndim != 2 or current_x.shape[0] != 1:
        msg = f'current_x must be shape (1, d), got {current_x.shape}'
        raise ValueError(msg)
    if current_x_model.shape != current_x.shape:
        msg = f'current_x_model must match current_x shape: got {current_x_model.shape}'
        raise ValueError(msg)

    d = current_x.shape[1]
    if proposal_chol.shape != (d, d):
        msg = f'proposal_chol must be shape ({d}, {d}), got {proposal_chol.shape}'
        raise ValueError(msg)

    # --- Propose new latent space sample ---
    proposal_x = current_x + (proposal_chol @ rng.standard_normal(d)).reshape(1, d)

    # --- Map to model space and evaluate ---
    proposal_x_model = sample_transformation_fn(proposal_x).reshape(1, d)
    loglike = log_likelihood_fn(proposal_x_model, simulation_number=step_num)
    logprior = log_prior_fn(proposal_x_model)
    logtarget = beta * loglike + logprior

    # --- Accept/reject step ---
    log_alpha = logtarget - logtarget_current
    if np.log(rng.uniform()) < log_alpha:
        return proposal_x, proposal_x_model, loglike, logtarget

    return current_x, current_x_model, loglike_current, logtarget_current


def run_mcmc_chain(
    initial_x,
    initial_x_model,
    loglike_initial,
    logtarget_initial,
    proposal_chol,
    beta,
    log_likelihood_fn,
    log_prior_fn,
    sample_transformation_fn,
    rng,
    num_steps,
    chain_num,
):
    """
    Run a single MCMC chain using fixed-scale Metropolis-Hastings steps.

    Parameters
    ----------
    initial_x : np.ndarray
        Initial sample in latent space, shape (1, d).
    initial_x_model : np.ndarray
        Corresponding model-space sample, shape (1, d).
    loglike_initial : float
        Log-likelihood of the initial model-space sample.
    logtarget_initial : float
        Log of the tempered target density at the initial sample.
    proposal_chol : np.ndarray
        Cholesky factor of the fixed proposal covariance matrix, shape (d, d).
    beta : float
        Current tempering parameter.
    log_likelihood_fn, log_prior_fn, sample_transformation_fn : callable
        Model functions.
    rng : np.random.Generator
        Random number generator.
    num_steps : int
        Number of Metropolis-Hastings steps to run.

    Returns
    -------
    final_x : np.ndarray
        Final latent space sample (1, d).
    final_x_model : np.ndarray
        Final model space sample (1, d).
    loglike : float
        Final log-likelihood.
    logtarget : float
        Final log-target density.
    """
    current_x = initial_x
    current_x_model = initial_x_model
    loglike_current = loglike_initial
    logtarget_current = logtarget_initial

    for _ in range(num_steps):
        current_x, current_x_model, loglike_current, logtarget_current = (
            metropolis_step(
                current_x,
                current_x_model,
                loglike_current,
                logtarget_current,
                proposal_chol,
                beta,
                rng,
                log_likelihood_fn,
                log_prior_fn,
                sample_transformation_fn,
                chain_num,
            )
        )

    return current_x, current_x_model, loglike_current, logtarget_current


def run_one_stage_equal_chain_lengths(
    samples,
    model_parameters,
    log_likelihood_values,
    beta,
    log_likelihood_fn,
    log_prior_fn,
    sample_transformation_fn,
    scale_factor,
    num_steps,
    proposal_cov_fn,
    seed=None,
    num_burn_in=0,
    thinning_factor=1,
    logger=None,
    run_type='runningLocal',
):
    """
    Run one TMCMC stage with equal-length MCMC chains and fixed proposal scale.

    Parameters
    ----------
    samples : np.ndarray
        Latent space samples from the previous TMCMC stage, shape (N, d).
    model_parameters : np.ndarray
        Corresponding model space samples, shape (N, d).
    log_likelihood_values : np.ndarray
        Log-likelihoods of model_parameters, shape (N,).
    beta : float
        Current tempering parameter.
    log_likelihood_fn, log_prior_fn, sample_transformation_fn : callable
        Model evaluation functions.
    scale_factor : float
        Scale multiplier for proposal covariance.
    num_steps : int
        Number of MH steps per MCMC chain.
    proposal_cov_fn : callable
        Function to compute proposal covariance: (samples, weights, scale) > cov.
    parallel_evaluation_fn : callable
        Function that evaluates run_mcmc_chain in parallel given (fn, job_args).
    seed : int or None
        Random seed for reproducibility.
    num_burn_in : int
        Number of burn-in MH steps (discarded internally, last sample is returned).
    thinning_factor : int
        If >1 and beta=1, the chain will run num_steps * thinning_factor steps after burn-in.
        Only the last sample is returned.

    Returns
    -------
    Tuple containing:
        - new_samples : np.ndarray of shape (N, d)
        - new_model_parameters : np.ndarray of shape (N, d)
        - new_log_likelihoods : np.ndarray of shape (N,)
        - new_log_target_densities : np.ndarray of shape (N,)
        - new_beta : float
        - log_evidence : float
    """
    num_samples = samples.shape[0]
    new_beta = _increment_beta(log_likelihood_values, beta)
    log_evidence = _calculate_log_evidence(new_beta - beta, log_likelihood_values)
    weights = _calculate_weights(new_beta - beta, log_likelihood_values)

    # Fixed proposal across all chains
    proposal_cov = proposal_cov_fn(samples, weights, scale_factor)
    proposal_chol = safer_cholesky.decompose(proposal_cov)

    # Seed setup
    ss = SeedSequence(seed)
    rng_seeds = ss.spawn(num_samples)
    chain_rngs = [default_rng(seed) for seed in rng_seeds]

    resampling_rng = default_rng(ss.spawn(1)[0])
    chain_starting_indices = resampling_rng.choice(
        num_samples, size=num_samples, p=weights.flatten()
    )

    chain_length = num_burn_in + num_steps
    if new_beta == 1 and thinning_factor > 1:
        chain_length = num_burn_in + int(num_steps * thinning_factor)
    total_num_model_evaluations = chain_length * num_samples
    if logger is not None:
        logger.info(f'    > Number of steps per chain = {chain_length}')
        logger.info(f'    > Number of chains = {num_samples}')
        logger.info(
            f'    > Number of model evaluations = {total_num_model_evaluations}'
        )

    # Build job arguments for each chain
    job_args = []
    for i, idx in enumerate(chain_starting_indices):
        x = np.atleast_2d(samples[idx]).reshape(1, -1)
        x_model = np.atleast_2d(model_parameters[idx]).reshape(1, -1)
        loglike = log_likelihood_values[idx].item()
        logprior = log_prior_fn(x_model)
        logtarget = new_beta * loglike + logprior

        job_args.append(
            (
                x,
                x_model,
                loglike,
                logtarget,
                proposal_chol,
                new_beta,
                log_likelihood_fn,
                log_prior_fn,
                sample_transformation_fn,
                chain_rngs[i],
                chain_length,
                i,
            )
        )
    parallel_runner = uq_utilities.get_parallel_pool_instance(run_type)
    if logger is not None:
        logger.info(
            f'    > Running {parallel_runner.num_processors} model evaluations in parallel'
        )
    results = parallel_runner.run(run_mcmc_chain, job_args)
    parallel_runner.close_pool()
    if logger is not None:
        logger.info('    > Model evaluations completed')

    # Gather results
    new_samples = np.zeros_like(samples)
    new_model_parameters = np.zeros_like(model_parameters)
    new_log_likelihoods = np.zeros_like(log_likelihood_values)
    new_log_target_densities = np.zeros_like(log_likelihood_values)

    for i, (x, x_model, loglike, logtarget) in enumerate(results):
        new_samples[i, :] = x.reshape(-1)
        new_model_parameters[i, :] = x_model.reshape(-1)
        new_log_likelihoods[i] = np.asarray(loglike).item()
        new_log_target_densities[i] = np.asarray(logtarget).item()

    return (
        new_samples,
        new_model_parameters,
        new_log_likelihoods,
        new_log_target_densities,
        new_beta,
        log_evidence,
        total_num_model_evaluations,
    )


def setup_logger(log_filename='logFileTMCMC.txt'):
    """
    Set up a logger for TMCMC execution that logs messages to a file.

    This function configures a logger named 'tmcmc' with INFO level logging
    and attaches a FileHandler to write logs to the specified file. If the
    logger already has handlers attached (e.g., due to repeated calls),
    it will not add duplicate handlers.

    Parameters
    ----------
    log_filename : str, optional
        The name of the log file to which messages will be written.
        Defaults to 'logFileTMCMC.txt'.

    Returns
    -------
    logging.Logger
        Configured logger instance ready for use.
    """
    logger = logging.getLogger('tmcmc')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_filename)
        fh.setFormatter(
            logging.Formatter('%(message)s')
            # logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(fh)
    return logger


class TMCMC:
    """
    A class to perform Transitional Markov Chain Monte Carlo (TMCMC) sampling.

    Attributes
    ----------
    _log_likelihood_function : callable
        Function to compute log-likelihoods.
    _log_prior_density_function : callable
        Function to compute log-prior densities.
    _sample_transformation_function : callable
        Function mapping latent-space samples to model-space.
    run_type : str, optional
            The run type ("runningLocal" or "runningRemote"). Defaults to "runningLocal".
    run_parallel : bool
        Whether to evaluate MCMC chains in parallel.
    num_steps : int
        Number of MCMC steps per chain.
    cov_threshold : float
        Target coefficient of variation threshold (currently unused).
    thinning_factor : int
        Thinning factor applied only at the final stage (if beta = 1).
    adapt_frequency : int
        Adaptation frequency for proposal scaling (used only in non-parallel mode).
    """

    def __init__(
        self,
        log_likelihood_function,
        log_prior_density_function,
        sample_transformation_function,
        seed=None,
        run_type='runningLocal',
        run_parallel=True,  # noqa: FBT002
        cov_threshold=1,
        num_steps=1,
        thinning_factor=10,
        adapt_frequency=50,
        log_filename='logFileTMCMC.txt',
    ):
        """
        Initialize the TMCMC class.

        Parameters
        ----------
        log_likelihood_function : callable
            Function to calculate log-likelihoods in model space.
        log_prior_density_function : callable
            Function to calculate log-prior densities in model space.
        sample_transformation_function : callable
            Function mapping latent-space samples (1, d) to model-space (1, d).
        seed : int or None, optional
            Seed for reproducibility.
        run_type : str, optional
            The run type ("runningLocal" or "runningRemote"). Defaults to "runningLocal".
        run_parallel : bool, optional
            Whether to run chains in parallel using parallel_evaluation_function. Defaults to True.
        cov_threshold : float, optional
            Coefficient of variation threshold (unused in current implementation). Defaults to 1.
        num_steps : int, optional
            Number of MCMC steps per chain. Defaults to 1.
        thinning_factor : int, optional
            Thinning factor for final stage chains. Defaults to 10.
        adapt_frequency : int, optional
            Adaptation frequency (used only in unequal-chain serial mode). Defaults to 50.
        """
        self._log_likelihood_function = log_likelihood_function
        self._log_prior_density_function = log_prior_density_function
        self._sample_transformation_function = sample_transformation_function
        self.run_type = run_type
        self.run_parallel = run_parallel
        self._seed = seed
        self.num_steps = num_steps
        self.cov_threshold = cov_threshold
        self.thinning_factor = thinning_factor
        self.adapt_frequency = adapt_frequency
        self.log_filename = log_filename

        self._logger = setup_logger(log_filename=self.log_filename)

    def flush_logs(self):
        for handler in self._logger.handlers:
            handler.flush()

    def run(
        self,
        samples_dict,
        model_parameters_dict,
        betas_dict,
        log_likelihood_values_dict,
        log_target_density_values_dict,
        log_evidence_dict,
        num_model_evals_dict,
        stage_num,
        num_burn_in=0,
    ):
        """
        Run the TMCMC algorithm.

        Parameters
        ----------
        samples_dict : dict
            Dictionary of latent-space samples at each stage.
        model_parameters_dict : dict
            Dictionary of model-space samples at each stage.
        betas_dict : dict
            Dictionary of beta values at each stage.
        log_likelihood_values_dict : dict
            Dictionary of log-likelihoods at each stage.
        log_target_density_values_dict : dict
            Dictionary of log-target densities at each stage.
        log_evidence_dict : dict
            Dictionary of log evidence estimates at each stage.
        stage_num : int
            Current TMCMC stage number.
        num_burn_in : int, optional
            Number of burn-in steps for MCMC chains. Defaults to 0.

        Returns
        -------
        dict
            Updated dictionaries containing TMCMC samples and log-values.
        """
        start_time = time.time()
        start_stage = stage_num
        if start_stage > 0:
            self._logger.info('Warm-starting TMCMC')
        else:
            self._logger.info('Starting TMCMC')
        self.num_dimensions = samples_dict[0].shape[1]
        seed_sequence = SeedSequence(self._seed)
        self.target_acceptance_rate = 0.23 + 0.21 / self.num_dimensions
        self.scale_factor = 2.4 / np.sqrt(self.num_dimensions)
        while betas_dict[stage_num] < 1:
            stage_start_time = time.time()
            self._logger.info(
                f'  Stage {stage_num} | Current β = {betas_dict[stage_num]:.4f}'
            )

            seed = seed_sequence.spawn(1)[0].entropy
            if self.run_parallel:
                (
                    new_samples,
                    new_model_parameters,
                    new_log_likelihood_values,
                    new_log_target_density_values,
                    new_beta,
                    log_evidence,
                    total_num_model_evaluations,
                ) = run_one_stage_equal_chain_lengths(
                    samples_dict[stage_num],
                    model_parameters_dict[stage_num],
                    log_likelihood_values_dict[stage_num],
                    betas_dict[stage_num],
                    self._log_likelihood_function,
                    self._log_prior_density_function,
                    self._sample_transformation_function,
                    self.scale_factor,
                    self.num_steps,
                    get_scaled_proposal_covariance,
                    seed=seed,
                    num_burn_in=num_burn_in,
                    thinning_factor=self.thinning_factor,
                    logger=self._logger,
                )
            else:
                (
                    new_samples,
                    new_model_parameters,
                    new_log_likelihood_values,
                    new_log_target_density_values,
                    new_beta,
                    log_evidence,
                    total_num_model_evaluations,
                ) = run_one_stage_unequal_chain_lengths(
                    samples_dict[stage_num],
                    model_parameters_dict[stage_num],
                    log_likelihood_values_dict[stage_num],
                    log_target_density_values_dict[stage_num],
                    betas_dict[stage_num],
                    default_rng(seed),
                    self._log_likelihood_function,
                    self._log_prior_density_function,
                    self._sample_transformation_function,
                    self.scale_factor,
                    self.target_acceptance_rate,
                    do_thinning=False,
                    burn_in_steps=num_burn_in,
                    number_of_steps=self.num_steps,
                    thinning_factor=self.thinning_factor,
                    adapt_frequency=self.adapt_frequency,
                    logger=self._logger,
                )
            stage_num += 1
            samples_dict[stage_num] = new_samples
            model_parameters_dict[stage_num] = new_model_parameters
            betas_dict[stage_num] = new_beta
            log_likelihood_values_dict[stage_num] = new_log_likelihood_values
            log_target_density_values_dict[stage_num] = new_log_target_density_values
            log_evidence_dict[stage_num] = log_evidence
            num_model_evals_dict[stage_num] = total_num_model_evaluations
            elapsed_time = time.time() - stage_start_time
            self._logger.info(
                f'    > New β = {new_beta:.4f}, log evidence increment = {log_evidence:.4f}'
            )
            self._logger.info(
                f'    > Time for this stage = {elapsed_time/60:.2f} minutes'
            )
            self._logger.info(' ')
            self.flush_logs()

        self._logger.info('TMCMC completed successfully.')
        self._logger.info(
            f'Total log-evidence: {sum(log_evidence_dict.values()):.4f}'
        )
        total_model_evaluations = sum(
            v for k, v in num_model_evals_dict.items() if k >= start_stage
        )
        self._logger.info(
            f'Total number of model evaluations: {total_model_evaluations}'
        )
        elapsed_time = time.time() - start_time
        self._logger.info(f'Total time: {elapsed_time/60:.2f} minutes')
        self._logger.info('-' * 45)
        self._logger.info(' ')

        return {
            'samples_dict': samples_dict,
            'model_parameters_dict': model_parameters_dict,
            'betas_dict': betas_dict,
            'log_likelihood_values_dict': log_likelihood_values_dict,
            'log_target_density_values_dict': log_target_density_values_dict,
            'log_evidence_dict': log_evidence_dict,
            'num_model_evals_dict': num_model_evals_dict,
        }


if __name__ == '__main__':
    import numpy as np
    import uq_utilities
    from tmcmc_test_utilities import (
        _log_likelihood_approximation_function,
        _log_prior_pdf,
        _sample_transformation_function,
    )

    # parallel_pool = uq_utilities.get_parallel_pool_instance('runningLocal')
    # parallel_evaluation_function = parallel_pool.pool.starmap
    # Initialize the TMCMC sampler
    tmcmc_sampler = TMCMC(
        _log_likelihood_approximation_function,
        _log_prior_pdf,
        _sample_transformation_function,
        run_parallel=True,
        run_type='runningLocal',
        cov_threshold=1,
        num_steps=1,
        thinning_factor=5,
        adapt_frequency=50,
    )

    # Initial parameters
    num_samples = 2000  # Number of samples
    num_dimensions = 2  # Dimensionality of the target distribution
    rng = np.random.default_rng(42)  # Random number generator

    # Start with some random samples
    initial_samples = rng.normal(size=(num_samples, num_dimensions))
    initial_model_parameters = _sample_transformation_function(initial_samples)
    initial_log_likelihoods = _log_likelihood_approximation_function(
        initial_model_parameters
    )
    initial_log_target_density_values = initial_log_likelihoods

    # Dictionaries to store results for each stage
    samples_dict = {0: initial_samples}
    model_parameters_dict = {0: initial_model_parameters}
    betas_dict = {0: 0.0}  # Start with beta=0 (prior importance)
    log_likelihoods_dict = {0: initial_log_likelihoods}
    log_target_density_values_dict = {0: initial_log_target_density_values}
    log_evidence_dict = {0: 0}
    num_model_evals_dict = {0: num_samples}

    # Run TMCMC
    stage_num = 0
    results = tmcmc_sampler.run(
        samples_dict,
        model_parameters_dict,
        betas_dict,
        log_likelihoods_dict,
        log_target_density_values_dict,
        log_evidence_dict,
        num_model_evals_dict,
        stage_num,
        num_burn_in=10,
    )

    # Unpack returned dictionary
    samples_dict = results['samples_dict']
    model_parameters_dict = results['model_parameters_dict']
    betas_dict = results['betas_dict']
    log_likelihoods_dict = results['log_likelihood_values_dict']
    log_target_density_values_dict = results['log_target_density_values_dict']
    log_evidence_dict = results['log_evidence_dict']
    num_model_evals_dict = results['num_model_evals_dict']

    # Display results
    final_stage_num = max(samples_dict.keys())
    print(  # noqa: T201
        f'Final samples (stage {final_stage_num}): \n{samples_dict[final_stage_num]}'
    )
    print(  # noqa: T201
        f'Final model parameters (stage {final_stage_num}): \n{model_parameters_dict[final_stage_num]}'
    )
    print(f'Betas: {betas_dict.values()}')  # noqa: T201
    print(f'Log-evidence values: {log_evidence_dict.values()}')  # noqa: T201
    print(f'Total log-evidence: {sum(log_evidence_dict.values())}')  # noqa: T201
    print(f'Number of model evaluations: {num_model_evals_dict.values()}')  # noqa: T201
