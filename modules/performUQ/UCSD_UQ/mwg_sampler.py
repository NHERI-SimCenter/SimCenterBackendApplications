import json  # noqa: CPY001, D100, INP001
from pathlib import Path

import numpy as np
import scipy

path_to_common_uq = Path(__file__).parent.parent / 'common'
import sys  # noqa: E402

sys.path.append(str(path_to_common_uq))
import uq_utilities  # noqa: E402


def _update_parameters_of_normal_inverse_wishart_distribution(  # noqa: ANN202
    niw_prior_parameters: uq_utilities.NormalInverseWishartParameters,
    n,  # noqa: ANN001
    theta_bar,  # noqa: ANN001
    s,  # noqa: ANN001
):
    lambda_0 = niw_prior_parameters.lambda_scalar
    mu_0 = niw_prior_parameters.mu_vector
    nu_0 = niw_prior_parameters.nu_scalar
    psi_0 = niw_prior_parameters.psi_matrix

    lambda_n = lambda_0 + n
    mu_n = (mu_0 * lambda_0 + n * theta_bar) / lambda_n
    nu_n = nu_0 + n
    psi_n = (
        psi_0
        + s
        + lambda_0 * n / lambda_n * ((theta_bar - mu_0) @ (theta_bar - mu_0).T)
    )
    return mu_n, lambda_n, nu_n, psi_n


def _update_parameters_of_inverse_gamma_distribution(  # noqa: ANN202
    inverse_gamma_prior_parameters: uq_utilities.InverseGammaParameters,
    n,  # noqa: ANN001
    sse,  # noqa: ANN001
):
    alpha_n = inverse_gamma_prior_parameters.alpha_scalar + n / 2
    beta_n = inverse_gamma_prior_parameters.beta_scalar + sse / 2
    return alpha_n, beta_n


def _draw_one_sample(  # noqa: ANN202, PLR0913, PLR0914, PLR0917
    sample_number,  # noqa: ANN001
    random_state,  # noqa: ANN001
    num_rv,  # noqa: ANN001
    num_edp,  # noqa: ANN001
    num_datasets,  # noqa: ANN001
    list_of_cholesky_decomposition_of_proposal_covariance_matrix,  # noqa: ANN001
    list_of_current_states,  # noqa: ANN001
    transformation_function,  # noqa: ANN001
    list_of_model_evaluation_functions,  # noqa: ANN001
    parallel_evaluation_function,  # noqa: ANN001
    function_to_evaluate,  # noqa: ANN001
    list_of_datasets,  # noqa: ANN001
    list_of_dataset_lengths,  # noqa: ANN001
    inverse_gamma_prior_parameters,  # noqa: ANN001
    list_of_unnormalized_posterior_logpdf_at_current_state,  # noqa: ANN001
    list_of_loglikelihood_at_current_state,  # noqa: ANN001
    list_of_prior_logpdf_at_current_state,  # noqa: ANN001
    niw_prior_parameters,  # noqa: ANN001
    loglikelihood_function,  # noqa: ANN001
    results_directory_path,  # noqa: ANN001
    tabular_results_file_base_name,  # noqa: ANN001
    current_mean_sample,  # noqa: ANN001
    current_cov_sample,  # noqa: ANN001
    list_of_current_error_variance_samples_scaled,  # noqa: ANN001
    num_accepts_list,  # noqa: ANN001
):
    prngs_rvs = uq_utilities.get_random_number_generators(
        entropy=(sample_number, random_state), num_prngs=num_rv
    )
    prngs_edps = uq_utilities.get_random_number_generators(
        entropy=(2 * sample_number, 2 * random_state), num_prngs=num_edp
    )
    prngs_algorithm = uq_utilities.get_random_number_generators(
        entropy=(3 * sample_number, 3 * random_state),
        num_prngs=num_datasets + 2,
    )
    iterable = []
    list_of_proposed_states = []
    for dataset_number in range(num_datasets):
        standard_normal_random_variates = (
            uq_utilities.get_standard_normal_random_variates(prngs_rvs)
        )
        cholesky_decomposition_of_proposal_covariance_matrix = (
            list_of_cholesky_decomposition_of_proposal_covariance_matrix[
                dataset_number
            ]
        )
        move = cholesky_decomposition_of_proposal_covariance_matrix @ np.array(
            standard_normal_random_variates
        ).reshape((-1, 1))
        current_state = np.array(list_of_current_states[dataset_number]).reshape(
            (-1, 1)
        )
        proposed_state = current_state + move
        x = transformation_function(proposed_state)
        model_iterable = [0, x]
        # model_iterable = [sample_number, x]
        inputs = [
            list_of_model_evaluation_functions[dataset_number],
            model_iterable,
        ]
        iterable.append(inputs)
        list_of_proposed_states.append(proposed_state)

    list_of_model_outputs = list(
        parallel_evaluation_function(function_to_evaluate, iterable)
    )

    list_of_sse = []
    list_of_log_likelihoods = []
    list_of_log_hastings_ratios = []
    list_of_new_states = []
    list_of_parameter_samples = []
    list_of_error_variance_samples = []
    list_of_error_variance_samples_scaled = []
    list_of_logpdf_of_proposed_states = []
    list_of_log_posteriors = []
    list_of_log_likes = []
    list_of_log_priors = []

    for dataset_number, dataset in enumerate(list_of_datasets):
        proposed_state = list_of_proposed_states[dataset_number]
        prior_logpdf_at_proposed_state = uq_utilities.multivariate_normal_logpdf(
            proposed_state,
            current_mean_sample,
            current_cov_sample,
        )
        list_of_logpdf_of_proposed_states.append(prior_logpdf_at_proposed_state)
        scaled_residual = (list_of_model_outputs[dataset_number] - dataset) / np.std(
            dataset
        )
        error_variance_sample_scaled = list_of_current_error_variance_samples_scaled[
            dataset_number
        ]
        log_likelihood_at_proposed_state = loglikelihood_function(
            scaled_residual,
            error_variance_sample_scaled,
        )
        unnormalized_posterior_logpdf_at_proposed_state = (
            log_likelihood_at_proposed_state + prior_logpdf_at_proposed_state
        )
        unnormalized_posterior_logpdf_at_current_state = (
            list_of_unnormalized_posterior_logpdf_at_current_state[dataset_number]
        )
        log_hastings_ratio = (
            unnormalized_posterior_logpdf_at_proposed_state
            - unnormalized_posterior_logpdf_at_current_state
        )
        list_of_log_hastings_ratios.append(log_hastings_ratio)
        list_of_log_likelihoods.append(log_likelihood_at_proposed_state)
        standard_uniform_random_variate = prngs_algorithm[dataset_number].uniform()
        proposed_state = list_of_proposed_states[dataset_number]
        current_state = list_of_current_states[dataset_number]
        if (log_hastings_ratio >= 0) | (
            np.log(standard_uniform_random_variate) < log_hastings_ratio
        ):  # accepted proposed state
            new_state = proposed_state
            list_of_log_posteriors.append(
                unnormalized_posterior_logpdf_at_proposed_state
            )
            list_of_log_likes.append(log_likelihood_at_proposed_state)
            list_of_log_priors.append(prior_logpdf_at_proposed_state)
            num_accepts_list[dataset_number] = num_accepts_list[dataset_number] + 1  # noqa: PLR6104
        else:
            new_state = current_state
            list_of_log_posteriors.append(
                unnormalized_posterior_logpdf_at_current_state
            )
            list_of_log_likes.append(
                list_of_loglikelihood_at_current_state[dataset_number]
            )
            list_of_log_priors.append(
                list_of_prior_logpdf_at_current_state[dataset_number]
            )
        new_state = np.array(new_state).reshape((-1, 1))
        list_of_new_states.append(new_state)
        list_of_parameter_samples.append(transformation_function(new_state).tolist())

        sse = scaled_residual @ scaled_residual
        list_of_sse.append(sse)
        n = list_of_dataset_lengths[dataset_number]
        alpha_n, beta_n = _update_parameters_of_inverse_gamma_distribution(
            inverse_gamma_prior_parameters=inverse_gamma_prior_parameters,
            n=n,
            sse=sse,
        )
        new_error_variance_sample_scaled = (
            uq_utilities.get_inverse_gamma_random_variate(
                prngs_edps[dataset_number], alpha_n, beta_n
            )
        ).item()
        list_of_error_variance_samples_scaled.append(
            new_error_variance_sample_scaled
        )
        new_error_variance_sample = (
            np.var(dataset) * new_error_variance_sample_scaled
        )
        list_of_error_variance_samples.append(new_error_variance_sample)

    n = num_datasets
    theta_bar = np.mean(
        np.array(list_of_new_states),
        axis=0,
    ).reshape((-1, 1))
    s = np.zeros((num_rv, num_rv))
    for new_state in list_of_new_states:
        s = s + (new_state - theta_bar) @ (new_state - theta_bar).T  # noqa: PLR6104
    mu_n, lambda_n, nu_n, psi_n = (
        _update_parameters_of_normal_inverse_wishart_distribution(
            niw_prior_parameters,
            n,
            theta_bar,
            s,
        )
    )
    updated_parameters = {}
    updated_parameters['mu_n'] = mu_n.flatten().tolist()
    updated_parameters['lambda_n'] = lambda_n
    updated_parameters['nu_n'] = nu_n
    updated_parameters['psi_n'] = psi_n.tolist()

    covariance_sample = scipy.stats.invwishart(
        df=nu_n,
        scale=psi_n,
    ).rvs(random_state=prngs_algorithm[-2])
    mean_sample = scipy.stats.multivariate_normal(
        mean=mu_n.flatten(),
        cov=covariance_sample / lambda_n,
    ).rvs(random_state=prngs_algorithm[-1])

    one_sample = {}
    one_sample['new_states'] = list_of_new_states
    one_sample['error_variances_scaled'] = list_of_error_variance_samples_scaled
    one_sample['hyper_covariance'] = covariance_sample
    one_sample['hyper_mean'] = mean_sample

    results_to_write = {}
    results_to_write['log_priors'] = list_of_logpdf_of_proposed_states
    results_to_write['log_likelihoods'] = list_of_log_likes
    results_to_write['unnormalized_log_posteriors'] = list_of_log_posteriors
    new_states_list = []
    for item in list_of_new_states:
        if isinstance(item, list):
            new_states_list.append(item)
        else:
            new_states_list.append(item.tolist())
    results_to_write['new_states'] = new_states_list
    results_to_write['error_variances_scaled'] = (
        list_of_error_variance_samples_scaled
    )
    results_to_write['hyper_covariance'] = covariance_sample.tolist()
    results_to_write['hyper_mean'] = mean_sample.tolist()
    results_to_write['updated_parameters_of_normal_inverse_wishart_distribution'] = (
        updated_parameters
    )

    for dataset_number in range(num_datasets):
        x = list_of_parameter_samples[dataset_number]
        x.append(list_of_error_variance_samples[dataset_number])
        y = list_of_model_outputs[dataset_number]
        list_of_strings_to_write = []
        list_of_strings_to_write.append(f'{sample_number + 1}')  # noqa: FURB113
        list_of_strings_to_write.append(f'{dataset_number + 1}')
        x_string_list = []
        for x_val in x:
            x_string_list.append(f'{x_val}')  # noqa: PERF401
        list_of_strings_to_write.append('\t'.join(x_string_list))
        y_string_list = []
        for y_val in y:
            y_string_list.append(f'{y_val}')  # noqa: PERF401
        list_of_strings_to_write.append('\t'.join(y_string_list))

        tabular_results_file_name = (
            uq_utilities._get_tabular_results_file_name_for_dataset(  # noqa: SLF001
                tabular_results_file_base_name, dataset_number
            )
        )
        string_to_write = '\t'.join(list_of_strings_to_write) + '\n'
        uq_utilities._write_to_tabular_results_file(  # noqa: SLF001
            tabular_results_file_name, string_to_write
        )

    with open(results_directory_path / f'sample_{sample_number + 1}.json', 'w') as f:  # noqa: PLW1514, PTH123
        json.dump(results_to_write, f, indent=4)

    return one_sample, results_to_write


def _get_tabular_results_file_name_for_hyperparameters(  # noqa: ANN202
    tabular_results_file_base_name,  # noqa: ANN001
):
    tabular_results_parent = tabular_results_file_base_name.parent
    tabular_results_stem = tabular_results_file_base_name.stem
    tabular_results_extension = tabular_results_file_base_name.suffix

    tabular_results_file = (
        tabular_results_parent
        / f'{tabular_results_stem}_hyperparameters{tabular_results_extension}'
    )
    return tabular_results_file  # noqa: RET504


def get_states_from_samples_list(samples_list, dataset_number):  # noqa: ANN001, ANN201, D103
    sample_values = []
    for sample_number in range(len(samples_list)):
        sample_values.append(  # noqa: PERF401
            samples_list[sample_number]['new_states'][dataset_number].flatten()
        )
    return sample_values


def tune(scale, acc_rate):  # noqa: ANN001, ANN201
    """Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:
    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """  # noqa: D205, D400
    if acc_rate < 0.01:  # noqa: PLR2004
        return scale * 0.01
    elif acc_rate < 0.05:  # noqa: RET505, PLR2004
        return scale * 0.1
    elif acc_rate < 0.2:  # noqa: PLR2004
        return scale * 0.5
    elif acc_rate > 0.95:  # noqa: PLR2004
        return scale * 100.0
    elif acc_rate > 0.75:  # noqa: PLR2004
        return scale * 10.0
    elif acc_rate > 0.5:  # noqa: PLR2004
        return scale * 2
    return scale


def metropolis_within_gibbs_sampler(  # noqa: ANN201, C901, D103, PLR0913, PLR0914, PLR0917
    uq_inputs,  # noqa: ANN001
    parallel_evaluation_function,  # noqa: ANN001
    function_to_evaluate,  # noqa: ANN001
    transformation_function,  # noqa: ANN001
    num_rv,  # noqa: ANN001
    num_edp,  # noqa: ANN001
    list_of_model_evaluation_functions,  # noqa: ANN001
    list_of_datasets,  # noqa: ANN001
    list_of_dataset_lengths,  # noqa: ANN001
    list_of_current_states,  # noqa: ANN001
    list_of_cholesky_of_proposal_covariance_matrix,  # noqa: ANN001
    inverse_gamma_prior_parameters,  # noqa: ANN001
    loglikelihood_function,  # noqa: ANN001
    list_of_unnormalized_posterior_logpdf_at_current_state,  # noqa: ANN001
    list_of_loglikelihood_at_current_state,  # noqa: ANN001
    list_of_prior_logpdf_at_current_state,  # noqa: ANN001
    niw_prior_parameters,  # noqa: ANN001
    results_directory_path,  # noqa: ANN001
    tabular_results_file_base_name,  # noqa: ANN001
    rv_inputs,  # noqa: ANN001
    edp_inputs,  # noqa: ANN001
    current_mean_sample,  # noqa: ANN001
    current_covariance_sample,  # noqa: ANN001
    list_of_current_error_variance_samples_scaled,  # noqa: ANN001
    parent_distribution,  # noqa: ANN001, ARG001
    num_accepts_list,  # noqa: ANN001
    proposal_scale_list,  # noqa: ANN001
    list_of_proposal_covariance_kernels,  # noqa: ANN001
):
    num_datasets = len(list_of_datasets)
    random_state = uq_inputs['Random State']
    tuning_interval = 200
    if 'Tuning Interval' in uq_inputs:
        tuning_interval = uq_inputs['Tuning Interval']
    tuning_period = 1000
    if 'Tuning Period' in uq_inputs:
        tuning_period = uq_inputs['Tuning Period']
    num_samples = uq_inputs['Sample Size'] + tuning_period
    parent_distribution_prng = (
        uq_utilities.get_list_of_pseudo_random_number_generators(
            10 * random_state, 1
        )[0]
    )

    initial_list_of_proposal_covariance_kernels = list_of_proposal_covariance_kernels  # noqa: F841

    for dataset_number in range(num_datasets):
        tabular_results_file_name = (
            uq_utilities._get_tabular_results_file_name_for_dataset(  # noqa: SLF001
                tabular_results_file_base_name, dataset_number
            )
        )
        rv_string_list = []
        for rv in rv_inputs:
            rv_string_list.append(rv['name'])  # noqa: PERF401
        error_var_string_list = []
        edp_string_list = []
        edp = edp_inputs[dataset_number]
        error_var_string_list.append(f'{edp["name"]}.PredictionErrorVariance')
        edp_components_list = []
        for edp_component in range(edp['length']):
            edp_components_list.append(f'{edp["name"]}_{edp_component + 1}')  # noqa: PERF401
        edp_string_list.append('\t'.join(edp_components_list))

        list_of_header_strings = []
        list_of_header_strings.append('eval_id')  # noqa: FURB113
        list_of_header_strings.append('interface')
        list_of_header_strings.append('\t'.join(rv_string_list))
        list_of_header_strings.append('\t'.join(error_var_string_list))
        list_of_header_strings.append('\t'.join(edp_string_list))
        string_to_write = '\t'.join(list_of_header_strings) + '\n'
        tabular_results_file_name.touch()
        uq_utilities._write_to_tabular_results_file(  # noqa: SLF001
            tabular_results_file_name, string_to_write
        )

    list_of_hyperparameter_header_strings = []
    list_of_hyperparameter_header_strings.append('eval_id')  # noqa: FURB113
    list_of_hyperparameter_header_strings.append('interface')
    rv_mean_string_list = []
    rv_names_list = []
    for rv in rv_inputs:
        rv_mean_string_list.append(f'mean_{rv["name"]}')
        rv_names_list.append(rv['name'])
    list_of_hyperparameter_header_strings.append('\t'.join(rv_mean_string_list))
    rv_covariance_string_list = []
    for i in range(len(rv_names_list)):
        for j in range(i, len(rv_names_list)):
            rv_covariance_string_list.append(  # noqa: PERF401
                f'cov_{rv_names_list[i]}_{rv_names_list[j]}'
            )
    list_of_hyperparameter_header_strings.append(
        '\t'.join(rv_covariance_string_list)
    )
    hyperparameter_header_string = (
        '\t'.join(list_of_hyperparameter_header_strings) + '\n'
    )
    hyperparameter_tabular_results_file_name = (
        _get_tabular_results_file_name_for_hyperparameters(
            tabular_results_file_base_name
        )
    )
    hyperparameter_tabular_results_file_name.touch()
    uq_utilities._write_to_tabular_results_file(  # noqa: SLF001
        hyperparameter_tabular_results_file_name,
        hyperparameter_header_string,
    )

    list_of_predictive_distribution_sample_header_strings = []
    list_of_predictive_distribution_sample_header_strings.append('eval_id')  # noqa: FURB113
    list_of_predictive_distribution_sample_header_strings.append('interface')

    list_of_predictive_distribution_sample_header_strings.append(
        '\t'.join(rv_names_list)
    )
    predictive_distribution_sample_header_string = (
        '\t'.join(list_of_predictive_distribution_sample_header_strings) + '\n'
    )
    tabular_results_file_base_name.touch()
    uq_utilities._write_to_tabular_results_file(  # noqa: SLF001
        tabular_results_file_base_name,
        predictive_distribution_sample_header_string,
    )

    samples = []
    for sample_number in range(num_samples):
        one_sample, results = _draw_one_sample(
            sample_number,
            random_state,
            num_rv,
            num_edp,
            num_datasets,
            list_of_cholesky_of_proposal_covariance_matrix,
            list_of_current_states,
            transformation_function,
            list_of_model_evaluation_functions,
            parallel_evaluation_function,
            function_to_evaluate,
            list_of_datasets,
            list_of_dataset_lengths,
            inverse_gamma_prior_parameters,
            list_of_unnormalized_posterior_logpdf_at_current_state,
            list_of_loglikelihood_at_current_state,
            list_of_prior_logpdf_at_current_state,
            niw_prior_parameters,
            loglikelihood_function,
            results_directory_path,
            tabular_results_file_base_name,
            current_mean_sample,
            current_covariance_sample,
            list_of_current_error_variance_samples_scaled,
            num_accepts_list,
        )
        samples.append(one_sample)
        list_of_current_states = one_sample['new_states']
        list_of_current_error_variance_samples_scaled = one_sample[
            'error_variances_scaled'
        ]
        current_mean_sample = one_sample['hyper_mean']
        current_covariance_sample = one_sample['hyper_covariance']

        list_of_unnormalized_posterior_logpdf_at_current_state = results[
            'unnormalized_log_posteriors'
        ]
        list_of_loglikelihood_at_current_state = results['log_likelihoods']
        list_of_prior_logpdf_at_current_state = results['log_priors']

        if (
            (sample_number >= tuning_interval)
            and (sample_number % tuning_interval == 0)
            and (sample_number <= tuning_period)
        ):
            list_of_acceptance_rates = []
            for dataset_number in range(num_datasets):
                num_accepts = num_accepts_list[dataset_number]
                acc_rate = num_accepts / tuning_interval
                list_of_acceptance_rates.append(acc_rate)
                proposal_scale = proposal_scale_list[dataset_number]
                proposal_scale = tune(proposal_scale, acc_rate)
                proposal_scale_list[dataset_number] = proposal_scale
                cov_kernel = list_of_proposal_covariance_kernels[dataset_number]
                if num_accepts > num_rv:
                    states = get_states_from_samples_list(samples, dataset_number)
                    samples_array = np.array(states[-tuning_interval:]).T
                    try:
                        cov_kernel = np.cov(samples_array)
                    except Exception as exc:  # noqa: BLE001
                        print(  # noqa: T201
                            f'Sample number: {sample_number}, dataset number:'
                            f' {dataset_number}, Exception in covariance'
                            f' calculation: {exc}'
                        )
                        cov_kernel = list_of_proposal_covariance_kernels[
                            dataset_number
                        ]
                proposal_covariance_matrix = cov_kernel * proposal_scale
                try:
                    cholesky_of_proposal_covariance_matrix = scipy.linalg.cholesky(
                        proposal_covariance_matrix, lower=True
                    )
                except Exception as exc:  # noqa: BLE001
                    print(  # noqa: T201
                        f'Sample number: {sample_number}, dataset number:'
                        f' {dataset_number}, Exception in cholesky'
                        f' calculation: {exc}'
                    )
                    cov_kernel = list_of_proposal_covariance_kernels[dataset_number]
                    proposal_covariance_matrix = cov_kernel * proposal_scale
                    cholesky_of_proposal_covariance_matrix = scipy.linalg.cholesky(
                        proposal_covariance_matrix, lower=True
                    )
                list_of_cholesky_of_proposal_covariance_matrix[dataset_number] = (
                    cholesky_of_proposal_covariance_matrix
                )
                list_of_proposal_covariance_kernels[dataset_number] = cov_kernel
            num_accepts_list = [0] * num_datasets

            adaptivity_results = {}
            adaptivity_results['list_of_acceptance_rates'] = list_of_acceptance_rates
            adaptivity_results['proposal_scale_list'] = proposal_scale_list
            cov_kernels_list = []
            for cov_kernel in list_of_proposal_covariance_kernels:
                cov_kernels_list.append(cov_kernel.tolist())  # noqa: PERF401
            adaptivity_results['list_of_proposal_covariance_kernels'] = (
                cov_kernels_list
            )
            with open(  # noqa: PLW1514, PTH123
                results_directory_path.parent
                / f'adaptivity_results_{sample_number}.json',
                'w',
            ) as f:
                json.dump(adaptivity_results, f, indent=4)

        hyper_mean_string_list = []
        hyper_mean = current_mean_sample
        for val in hyper_mean:
            hyper_mean_string_list.append(f'{val}')  # noqa: PERF401
        hyper_covariance_string_list = []
        hyper_covariance = current_covariance_sample
        for i in range(len(rv_names_list)):
            for j in range(i, len(rv_names_list)):
                hyper_covariance_string_list.append(f'{hyper_covariance[i][j]}')  # noqa: PERF401
        list_of_hyperparameter_value_strings = []
        list_of_hyperparameter_value_strings.append(f'{sample_number + 1}')  # noqa: FURB113
        list_of_hyperparameter_value_strings.append('0')
        list_of_hyperparameter_value_strings.append(
            '\t'.join(hyper_mean_string_list)
        )
        list_of_hyperparameter_value_strings.append(
            '\t'.join(hyper_covariance_string_list)
        )
        hyperparameter_value_string = (
            '\t'.join(list_of_hyperparameter_value_strings) + '\n'
        )
        uq_utilities._write_to_tabular_results_file(  # noqa: SLF001
            hyperparameter_tabular_results_file_name,
            hyperparameter_value_string,
        )

    # Get mean of updated predictive distribution parameters
    mu_n = []
    lambda_n = []
    nu_n = []
    psi_n = []
    n_samples_for_mean_of_updated_predictive_distribution_parameters = int(
        1 / 5 * num_samples
    )
    for i in range(
        num_samples
        - n_samples_for_mean_of_updated_predictive_distribution_parameters,
        num_samples,
    ):
        with open(results_directory_path / f'sample_{i + 1}.json') as f:  # noqa: PLW1514, PTH123
            data = json.load(f)
            updated_parameters = data[
                'updated_parameters_of_normal_inverse_wishart_distribution'
            ]
            mu_n.append(updated_parameters['mu_n'])
            lambda_n.append(updated_parameters['lambda_n'])
            nu_n.append(updated_parameters['nu_n'])
            psi_n.append(updated_parameters['psi_n'])

    mu_n_mean = np.mean(np.array(mu_n), axis=0)
    lambda_n_mean = np.mean(np.array(lambda_n), axis=0)
    nu_n_mean = np.mean(np.array(nu_n), axis=0)
    psi_n_mean = np.mean(np.array(psi_n), axis=0)

    df = nu_n_mean - num_datasets + 1  # noqa: PD901
    loc = mu_n_mean
    shape = (lambda_n_mean + 1) / (lambda_n_mean * df) * psi_n_mean
    predictive_distribution = scipy.stats.multivariate_t(loc=loc, shape=shape, df=df)
    for sample_number in range(num_samples):
        sample_from_predictive_t_distribution = predictive_distribution.rvs(
            random_state=parent_distribution_prng
        )
        sample_from_predictive_distribution = transformation_function(
            sample_from_predictive_t_distribution
        )
        while np.sum(np.isfinite(sample_from_predictive_distribution)) < num_rv:
            sample_from_predictive_t_distribution = predictive_distribution.rvs(
                random_state=parent_distribution_prng
            )
            sample_from_predictive_distribution = transformation_function(
                sample_from_predictive_t_distribution
            )
        predictive_distribution_sample_values_list = []
        for val in sample_from_predictive_distribution:
            predictive_distribution_sample_values_list.append(f'{val}')  # noqa: PERF401
        list_of_predictive_distribution_sample_value_strings = []
        list_of_predictive_distribution_sample_value_strings.append(  # noqa: FURB113
            f'{sample_number + 1}'
        )
        list_of_predictive_distribution_sample_value_strings.append('0')
        list_of_predictive_distribution_sample_value_strings.append(
            '\t'.join(predictive_distribution_sample_values_list)
        )
        predictive_distribution_sample_value_string = (
            '\t'.join(list_of_predictive_distribution_sample_value_strings) + '\n'
        )
        uq_utilities._write_to_tabular_results_file(  # noqa: SLF001
            tabular_results_file_base_name,
            predictive_distribution_sample_value_string,
        )

    return samples
