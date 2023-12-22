import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.stats

import preprocess_hierarchical_bayesian

path_to_common_uq = Path(__file__).parent.parent / "common"
sys.path.append(str(path_to_common_uq))
import uq_utilities


@dataclass
class NormalInverseWishartParameters:
    mu_vector: npt.NDArray
    lambda_scalar: float
    nu_scalar: float
    psi_matrix: npt.NDArray


@dataclass
class InverseGammaParameters:
    alpha_scalar: float
    beta_scalar: float

    def _to_shape_and_scale(self):
        return (self.alpha_scalar, 1 / self.beta_scalar)


def _get_tabular_results_file_name_for_hyperparameters(
    tabular_results_file_base_name,
):
    tabular_results_parent = tabular_results_file_base_name.parent
    tabular_results_stem = tabular_results_file_base_name.stem
    tabular_results_extension = tabular_results_file_base_name.suffix

    tabular_results_file = (
        tabular_results_parent
        / f"{tabular_results_stem}_hyperparameters{tabular_results_extension}"
    )
    return tabular_results_file


def _get_tabular_results_file_name_for_dataset(
    tabular_results_file_base_name, dataset_number
):
    tabular_results_parent = tabular_results_file_base_name.parent
    tabular_results_stem = tabular_results_file_base_name.stem
    tabular_results_extension = tabular_results_file_base_name.suffix

    tabular_results_file = (
        tabular_results_parent
        / f"{tabular_results_stem}_dataset_{dataset_number+1}{tabular_results_extension}"
    )
    return tabular_results_file


def _write_to_tabular_results_file(tabular_results_file, string_to_write):
    with tabular_results_file.open("a") as f:
        f.write(string_to_write)


def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:
    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    if acc_rate < 0.01:
        return scale * 0.01
    elif acc_rate < 0.05:
        return scale * 0.1
    elif acc_rate < 0.2:
        return scale * 0.5
    elif acc_rate > 0.95:
        return scale * 100.0
    elif acc_rate > 0.75:
        return scale * 10.0
    elif acc_rate > 0.5:
        return scale * 2
    return scale


def get_states_from_samples_list(samples_list, dataset_number):
    sample_values = []
    for sample_number in range(len(samples_list)):
        sample_values.append(
            samples_list[sample_number]["new_states"][dataset_number].flatten()
        )
    return sample_values


def _draw_one_sample(
    sample_number,
    random_state,
    num_rv,
    num_edp,
    num_datasets,
    list_of_cholesky_decomposition_of_proposal_covariance_matrix,
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
    current_cov_sample,
    list_of_current_error_variance_samples_scaled,
    num_accepts_list,
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
        current_state = np.array(
            list_of_current_states[dataset_number]
        ).reshape((-1, 1))
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

    list_of_model_outputs = parallel_evaluation_function(
        func=function_to_evaluate, iterable=iterable
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
        prior_logpdf_at_proposed_state = (
            uq_utilities.multivariate_normal_logpdf(
                proposed_state,
                current_mean_sample,
                current_cov_sample,
            )
        )
        list_of_logpdf_of_proposed_states.append(
            prior_logpdf_at_proposed_state
        )
        scaled_residual = (
            list_of_model_outputs[dataset_number] - dataset
        ) / np.std(dataset)
        error_variance_sample_scaled = (
            list_of_current_error_variance_samples_scaled[dataset_number]
        )
        log_likelihood_at_proposed_state = loglikelihood_function(
            scaled_residual,
            error_variance_sample_scaled,
        )
        unnormalized_posterior_logpdf_at_proposed_state = (
            log_likelihood_at_proposed_state + prior_logpdf_at_proposed_state
        )
        unnormalized_posterior_logpdf_at_current_state = (
            list_of_unnormalized_posterior_logpdf_at_current_state[
                dataset_number
            ]
        )
        log_hastings_ratio = (
            unnormalized_posterior_logpdf_at_proposed_state
            - unnormalized_posterior_logpdf_at_current_state
        )
        list_of_log_hastings_ratios.append(log_hastings_ratio)
        list_of_log_likelihoods.append(log_likelihood_at_proposed_state)
        standard_uniform_random_variate = prngs_algorithm[
            dataset_number
        ].uniform()
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
            num_accepts_list[dataset_number] = (
                num_accepts_list[dataset_number] + 1
            )
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
        list_of_parameter_samples.append(
            transformation_function(new_state).tolist()
        )

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
        s = s + (new_state - theta_bar) @ (new_state - theta_bar).T
    mu_n, lambda_n, nu_n, psi_n = (
        _update_parameters_of_normal_inverse_wishart_distribution(
            niw_prior_parameters,
            n,
            theta_bar,
            s,
        )
    )
    updated_parameters = {}
    updated_parameters["mu_n"] = mu_n.flatten().tolist()
    updated_parameters["lambda_n"] = lambda_n
    updated_parameters["nu_n"] = nu_n
    updated_parameters["psi_n"] = psi_n.tolist()

    covariance_sample = scipy.stats.invwishart(
        df=nu_n,
        scale=psi_n,
    ).rvs(random_state=prngs_algorithm[-2])
    mean_sample = scipy.stats.multivariate_normal(
        mean=mu_n.flatten(),
        cov=covariance_sample / lambda_n,
    ).rvs(random_state=prngs_algorithm[-1])

    one_sample = {}
    one_sample["new_states"] = list_of_new_states
    one_sample["error_variances_scaled"] = (
        list_of_error_variance_samples_scaled
    )
    one_sample["hyper_covariance"] = covariance_sample
    one_sample["hyper_mean"] = mean_sample

    results_to_write = {}
    results_to_write["log_priors"] = list_of_logpdf_of_proposed_states
    results_to_write["log_likelihoods"] = list_of_log_likes
    results_to_write["unnormalized_log_posteriors"] = list_of_log_posteriors
    new_states_list = []
    for item in list_of_new_states:
        if isinstance(item, list):
            new_states_list.append(item)
        else:
            new_states_list.append(item.tolist())
    results_to_write["new_states"] = new_states_list
    results_to_write["error_variances_scaled"] = (
        list_of_error_variance_samples_scaled
    )
    results_to_write["hyper_covariance"] = covariance_sample.tolist()
    results_to_write["hyper_mean"] = mean_sample.tolist()
    results_to_write[
        "updated_parameters_of_normal_inverse_wishart_distribution"
    ] = updated_parameters

    for dataset_number in range(num_datasets):
        x = list_of_parameter_samples[dataset_number]
        x.append(list_of_error_variance_samples[dataset_number])
        y = list_of_model_outputs[dataset_number]
        list_of_strings_to_write = []
        list_of_strings_to_write.append(f"{sample_number+1}")
        list_of_strings_to_write.append(f"{dataset_number+1}")
        x_string_list = []
        for x_val in x:
            x_string_list.append(f"{x_val}")
        list_of_strings_to_write.append("\t".join(x_string_list))
        y_string_list = []
        for y_val in y:
            y_string_list.append(f"{y_val}")
        list_of_strings_to_write.append("\t".join(y_string_list))

        tabular_results_file_name = _get_tabular_results_file_name_for_dataset(
            tabular_results_file_base_name, dataset_number
        )
        string_to_write = "\t".join(list_of_strings_to_write) + "\n"
        _write_to_tabular_results_file(
            tabular_results_file_name, string_to_write
        )

    with open(
        results_directory_path / f"sample_{sample_number+1}.json", "w"
    ) as f:
        json.dump(results_to_write, f, indent=4)

    return one_sample, results_to_write


def _update_parameters_of_normal_inverse_wishart_distribution(
    niw_prior_parameters: NormalInverseWishartParameters,
    n,
    theta_bar,
    s,
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


def _update_parameters_of_inverse_gamma_distribution(
    inverse_gamma_prior_parameters: InverseGammaParameters,
    n,
    sse,
):
    alpha_n = inverse_gamma_prior_parameters.alpha_scalar + n / 2
    beta_n = inverse_gamma_prior_parameters.beta_scalar + sse / 2
    return alpha_n, beta_n


def metropolis_within_gibbs_sampler(
    uq_inputs,
    parallel_evaluation_function,
    function_to_evaluate,
    transformation_function,
    num_rv,
    num_edp,
    list_of_model_evaluation_functions,
    list_of_datasets,
    list_of_dataset_lengths,
    list_of_current_states,
    list_of_cholesky_of_proposal_covariance_matrix,
    inverse_gamma_prior_parameters,
    loglikelihood_function,
    list_of_unnormalized_posterior_logpdf_at_current_state,
    list_of_loglikelihood_at_current_state,
    list_of_prior_logpdf_at_current_state,
    niw_prior_parameters,
    results_directory_path,
    tabular_results_file_base_name,
    rv_inputs,
    edp_inputs,
    current_mean_sample,
    current_covariance_sample,
    list_of_current_error_variance_samples_scaled,
    parent_distribution,
    num_accepts_list,
    proposal_scale_list,
    list_of_proposal_covariance_kernels,
):
    num_datasets = len(list_of_datasets)
    random_state = uq_inputs["Random State"]
    tuning_interval = 200
    if "Tuning Interval" in uq_inputs:
        tuning_interval = uq_inputs["Tuning Interval"]
    tuning_period = 1000
    if "Tuning Period" in uq_inputs:
        tuning_period = uq_inputs["Tuning Period"]
    num_samples = uq_inputs["Sample Size"] + tuning_period
    parent_distribution_prng = (
        uq_utilities.get_list_of_pseudo_random_number_generators(
            10 * random_state, 1
        )[0]
    )

    initial_list_of_proposal_covariance_kernels = (
        list_of_proposal_covariance_kernels
    )

    for dataset_number in range(num_datasets):
        tabular_results_file_name = _get_tabular_results_file_name_for_dataset(
            tabular_results_file_base_name, dataset_number
        )
        rv_string_list = []
        for rv in rv_inputs:
            rv_string_list.append(rv["name"])
        error_var_string_list = []
        edp_string_list = []
        edp = edp_inputs[dataset_number]
        error_var_string_list.append(f'{edp["name"]}.PredictionErrorVariance')
        edp_components_list = []
        for edp_component in range(edp["length"]):
            edp_components_list.append(f'{edp["name"]}_{edp_component+1}')
        edp_string_list.append("\t".join(edp_components_list))

        list_of_header_strings = []
        list_of_header_strings.append("eval_id")
        list_of_header_strings.append("interface")
        list_of_header_strings.append("\t".join(rv_string_list))
        list_of_header_strings.append("\t".join(error_var_string_list))
        list_of_header_strings.append("\t".join(edp_string_list))
        string_to_write = "\t".join(list_of_header_strings) + "\n"
        tabular_results_file_name.touch()
        _write_to_tabular_results_file(
            tabular_results_file_name, string_to_write
        )

    list_of_hyperparameter_header_strings = []
    list_of_hyperparameter_header_strings.append("eval_id")
    list_of_hyperparameter_header_strings.append("interface")
    rv_mean_string_list = []
    rv_names_list = []
    for rv in rv_inputs:
        rv_mean_string_list.append(f'mean_{rv["name"]}')
        rv_names_list.append(rv["name"])
    list_of_hyperparameter_header_strings.append(
        "\t".join(rv_mean_string_list)
    )
    rv_covariance_string_list = []
    for i in range(len(rv_names_list)):
        for j in range(i, len(rv_names_list)):
            rv_covariance_string_list.append(
                f"cov_{rv_names_list[i]}_{rv_names_list[j]}"
            )
    list_of_hyperparameter_header_strings.append(
        "\t".join(rv_covariance_string_list)
    )
    hyperparameter_header_string = (
        "\t".join(list_of_hyperparameter_header_strings) + "\n"
    )
    hyperparameter_tabular_results_file_name = (
        _get_tabular_results_file_name_for_hyperparameters(
            tabular_results_file_base_name
        )
    )
    hyperparameter_tabular_results_file_name.touch()
    _write_to_tabular_results_file(
        hyperparameter_tabular_results_file_name,
        hyperparameter_header_string,
    )

    list_of_predictive_distribution_sample_header_strings = []
    list_of_predictive_distribution_sample_header_strings.append("eval_id")
    list_of_predictive_distribution_sample_header_strings.append("interface")

    list_of_predictive_distribution_sample_header_strings.append(
        "\t".join(rv_names_list)
    )
    predictive_distribution_sample_header_string = (
        "\t".join(list_of_predictive_distribution_sample_header_strings) + "\n"
    )
    tabular_results_file_base_name.touch()
    _write_to_tabular_results_file(
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
        list_of_current_states = one_sample["new_states"]
        list_of_current_error_variance_samples_scaled = one_sample[
            "error_variances_scaled"
        ]
        current_mean_sample = one_sample["hyper_mean"]
        current_covariance_sample = one_sample["hyper_covariance"]

        list_of_unnormalized_posterior_logpdf_at_current_state = results[
            "unnormalized_log_posteriors"
        ]
        list_of_loglikelihood_at_current_state = results["log_likelihoods"]
        list_of_prior_logpdf_at_current_state = results["log_priors"]

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
                cov_kernel = list_of_proposal_covariance_kernels[
                    dataset_number
                ]
                if num_accepts > num_rv:
                    states = get_states_from_samples_list(
                        samples, dataset_number
                    )
                    samples_array = np.array(states[-tuning_interval:]).T
                    try:
                        cov_kernel = np.cov(samples_array)
                    except Exception as exc:
                        print(
                            f"Sample number: {sample_number}, dataset number:"
                            f" {dataset_number}, Exception in covariance"
                            f" calculation: {exc}"
                        )
                        cov_kernel = list_of_proposal_covariance_kernels[
                            dataset_number
                        ]
                proposal_covariance_matrix = cov_kernel * proposal_scale
                try:
                    cholesky_of_proposal_covariance_matrix = (
                        scipy.linalg.cholesky(
                            proposal_covariance_matrix, lower=True
                        )
                    )
                except Exception as exc:
                    print(
                        f"Sample number: {sample_number}, dataset number:"
                        f" {dataset_number}, Exception in cholesky"
                        f" calculation: {exc}"
                    )
                    cov_kernel = list_of_proposal_covariance_kernels[
                        dataset_number
                    ]
                    proposal_covariance_matrix = cov_kernel * proposal_scale
                    cholesky_of_proposal_covariance_matrix = (
                        scipy.linalg.cholesky(
                            proposal_covariance_matrix, lower=True
                        )
                    )
                list_of_cholesky_of_proposal_covariance_matrix[
                    dataset_number
                ] = cholesky_of_proposal_covariance_matrix
                list_of_proposal_covariance_kernels[dataset_number] = (
                    cov_kernel
                )
            num_accepts_list = [0] * num_datasets

            adaptivity_results = {}
            adaptivity_results["list_of_acceptance_rates"] = (
                list_of_acceptance_rates
            )
            adaptivity_results["proposal_scale_list"] = proposal_scale_list
            cov_kernels_list = []
            for cov_kernel in list_of_proposal_covariance_kernels:
                cov_kernels_list.append(cov_kernel.tolist())
            adaptivity_results["list_of_proposal_covariance_kernels"] = (
                cov_kernels_list
            )
            with open(
                results_directory_path.parent
                / f"adaptivity_results_{sample_number}.json",
                "w",
            ) as f:
                json.dump(adaptivity_results, f, indent=4)

        hyper_mean_string_list = []
        hyper_mean = current_mean_sample
        for val in hyper_mean:
            hyper_mean_string_list.append(f"{val}")
        hyper_covariance_string_list = []
        hyper_covariance = current_covariance_sample
        for i in range(len(rv_names_list)):
            for j in range(i, len(rv_names_list)):
                hyper_covariance_string_list.append(
                    f"{hyper_covariance[i][j]}"
                )
        list_of_hyperparameter_value_strings = []
        list_of_hyperparameter_value_strings.append(f"{sample_number+1}")
        list_of_hyperparameter_value_strings.append("0")
        list_of_hyperparameter_value_strings.append(
            "\t".join(hyper_mean_string_list)
        )
        list_of_hyperparameter_value_strings.append(
            "\t".join(hyper_covariance_string_list)
        )
        hyperparameter_value_string = (
            "\t".join(list_of_hyperparameter_value_strings) + "\n"
        )
        _write_to_tabular_results_file(
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
        with open(results_directory_path / f"sample_{i+1}.json", "r") as f:
            data = json.load(f)
            updated_parameters = data[
                "updated_parameters_of_normal_inverse_wishart_distribution"
            ]
            mu_n.append(updated_parameters["mu_n"])
            lambda_n.append(updated_parameters["lambda_n"])
            nu_n.append(updated_parameters["nu_n"])
            psi_n.append(updated_parameters["psi_n"])

    mu_n_mean = np.mean(np.array(mu_n), axis=0)
    lambda_n_mean = np.mean(np.array(lambda_n), axis=0)
    nu_n_mean = np.mean(np.array(nu_n), axis=0)
    psi_n_mean = np.mean(np.array(psi_n), axis=0)

    df = nu_n_mean - num_datasets + 1
    loc = mu_n_mean
    shape = (lambda_n_mean + 1) / (lambda_n_mean * df) * psi_n_mean
    predictive_distribution = scipy.stats.multivariate_t(
        loc=loc, shape=shape, df=df
    )
    for sample_number in range(num_samples):
        sample_from_predictive_t_distribution = predictive_distribution.rvs(
            random_state=parent_distribution_prng
        )
        sample_from_predictive_distribution = transformation_function(
            sample_from_predictive_t_distribution
        )
        while (
            np.sum(np.isfinite(sample_from_predictive_distribution)) < num_rv
        ):
            sample_from_predictive_t_distribution = (
                predictive_distribution.rvs(
                    random_state=parent_distribution_prng
                )
            )
            sample_from_predictive_distribution = transformation_function(
                sample_from_predictive_t_distribution
            )
        predictive_distribution_sample_values_list = []
        for val in sample_from_predictive_distribution:
            predictive_distribution_sample_values_list.append(f"{val}")
        list_of_predictive_distribution_sample_value_strings = []
        list_of_predictive_distribution_sample_value_strings.append(
            f"{sample_number+1}"
        )
        list_of_predictive_distribution_sample_value_strings.append("0")
        list_of_predictive_distribution_sample_value_strings.append(
            "\t".join(predictive_distribution_sample_values_list)
        )
        predictive_distribution_sample_value_string = (
            "\t".join(list_of_predictive_distribution_sample_value_strings)
            + "\n"
        )
        _write_to_tabular_results_file(
            tabular_results_file_base_name,
            predictive_distribution_sample_value_string,
        )

    return samples


def generate_initial_states(
    num_edp,
    num_rv,
    num_datasets,
    restart_file,
):
    list_of_initial_states_of_model_parameters = [
        np.zeros((num_rv, 1)) for _ in range(num_datasets)
    ]
    list_of_initial_states_of_error_variance_per_dataset = [
        0.1**2 for _ in range(num_edp)
    ]
    initial_state_of_hypermean = np.zeros((num_rv, 1))
    initial_state_of_hypercovariance = 0.2 * np.eye(num_rv)

    if restart_file is not None:
        restart_file_path = Path(restart_file)
        with restart_file_path.open("r") as f:
            restart_data = json.load(f)
        if "new_states" in restart_data:
            list_of_initial_states_of_model_parameters = []
            states_list = restart_data["new_states"]
            for state in states_list:
                list_of_initial_states_of_model_parameters.append(
                    np.array(state).reshape((num_rv, 1))
                )
        if "error_variances_scaled" in restart_data:
            list_of_initial_states_of_error_variance_per_dataset = (
                restart_data["error_variances_scaled"]
            )
        if "hyper_covariance" in restart_data:
            initial_state_of_hypercovariance = np.array(
                restart_data["hyper_covariance"]
            )
        if "hyper_mean" in restart_data:
            initial_state_of_hypermean = np.array(restart_data["hyper_mean"])

    return (
        list_of_initial_states_of_model_parameters,
        list_of_initial_states_of_error_variance_per_dataset,
        initial_state_of_hypermean,
        initial_state_of_hypercovariance,
    )


def loglikelihood_function(residual, error_variance_sample):
    mean = 0
    var = error_variance_sample
    standard_deviation = np.sqrt(var)
    ll = np.sum(scipy.stats.norm(mean, standard_deviation).logpdf(residual))
    if np.isnan(ll):
        ll = -np.inf
    return ll


def main(input_args):
    # Initialize analysis
    main_script_directory = Path(input_args[0]).resolve().parent
    working_directory = Path(input_args[1]).resolve()
    template_directory = Path(input_args[2]).resolve()
    run_type = input_args[3]  # either "runningLocal" or "runningRemote"
    workflow_driver = input_args[4]
    input_file = input_args[5]

    with open(input_file, "r") as f:
        inputs = json.load(f)

    uq_inputs = inputs["UQ"]
    rv_inputs = inputs["randomVariables"]
    edp_inputs = inputs["EDP"]

    (
        parallel_evaluation_function,
        function_to_evaluate,
        joint_distribution,
        num_rv,
        num_edp,
        list_of_model_evaluation_functions,
        list_of_datasets,
        list_of_dataset_lengths,
        restart_file,
    ) = preprocess_hierarchical_bayesian.preprocess_arguments(input_args)
    transformation_function = joint_distribution.u_to_x

    prior_inverse_gamma_parameters = InverseGammaParameters(
        alpha_scalar=1 / 2, beta_scalar=1 / 2
    )

    prior_normal_inverse_wishart_parameters = NormalInverseWishartParameters(
        mu_vector=np.zeros((num_rv, 1)),
        lambda_scalar=0,
        nu_scalar=num_rv + 2,
        psi_matrix=np.eye(num_rv),
    )

    num_datasets = len(list_of_datasets)
    (
        list_of_initial_states_of_model_parameters,
        list_of_initial_states_of_error_variance_per_dataset,
        initial_state_of_hypermean,
        initial_state_of_hypercovariance,
    ) = generate_initial_states(
        num_edp,
        num_rv,
        num_datasets,
        restart_file,
    )

    # TODO: get_initial_states():
    # either:
    # read them from file or
    # use LHS to explore the space and find the best starting points out of
    # those sampled values for the different chains
    # TODO: get_initial_proposal_covariance_matrix():
    # either:
    # read them from file or
    # adaptively tune the proposal covariance matrix by running the chain for
    # several thousand steps and keeping track of the acceptance rate
    # After every few hundred steps, calculate the covariance of the last few
    # hundred states of the chain, and adjust the scale factor multiplying the
    # covariance matrix of the last few hundred steps to keep the acceptance
    # rate within 20%-40%

    num_accepts_list = [0] * num_datasets
    # scale = np.square(1 / 50)
    scale = 2.38**2 / num_rv
    proposal_scale_list = [scale] * num_datasets

    cov_kernel = (1 / 50) ** 2 * np.eye(num_rv)

    list_of_proposal_covariance_kernels = []
    list_of_cholesky_of_proposal_covariance_matrix = []
    for dataset_number in range(num_datasets):
        proposal_covariance_matrix = (
            proposal_scale_list[dataset_number] * cov_kernel
        )
        list_of_proposal_covariance_kernels.append(cov_kernel)

        cholesky_of_proposal_covariance_matrix = scipy.linalg.cholesky(
            proposal_covariance_matrix, lower=True
        )
        list_of_cholesky_of_proposal_covariance_matrix.append(
            cholesky_of_proposal_covariance_matrix
        )

    parent_distribution = scipy.stats.multivariate_normal

    list_of_prior_logpdf_values = []
    iterable = []
    for model_number in range(len(list_of_datasets)):
        initial_state = list_of_initial_states_of_model_parameters[
            model_number
        ]
        x = transformation_function(initial_state)
        logpdf_of_initial_state = uq_utilities.multivariate_normal_logpdf(
            initial_state,
            initial_state_of_hypermean,
            initial_state_of_hypercovariance,
        )
        list_of_prior_logpdf_values.append(logpdf_of_initial_state)
        model_iterable = [0, x]
        # model_iterable = [sample_number, x]
        inputs = [
            list_of_model_evaluation_functions[model_number],
            model_iterable,
        ]
        iterable.append(inputs)

    list_of_model_outputs = parallel_evaluation_function(
        func=function_to_evaluate, iterable=iterable
    )

    list_of_unnormalized_posterior_logpdf_at_initial_state = []
    list_of_loglikelihood_at_initial_state = []
    list_of_prior_logpdf_at_initial_state = []
    for dataset_number, dataset in enumerate(list_of_datasets):
        scaled_residual = (
            list_of_model_outputs[dataset_number] - dataset
        ) / np.std(dataset)
        error_variance_sample_scaled = (
            list_of_initial_states_of_error_variance_per_dataset[
                dataset_number
            ]
        )
        log_likelihood_at_initial_state = loglikelihood_function(
            scaled_residual,
            error_variance_sample_scaled,
        )
        prior_logpdf_at_initial_state = list_of_prior_logpdf_values[
            dataset_number
        ]
        unnormalized_posterior_logpdf_at_initial_state = (
            log_likelihood_at_initial_state + prior_logpdf_at_initial_state
        )
        list_of_unnormalized_posterior_logpdf_at_initial_state.append(
            unnormalized_posterior_logpdf_at_initial_state
        )
        list_of_loglikelihood_at_initial_state.append(
            log_likelihood_at_initial_state
        )
        list_of_prior_logpdf_at_initial_state.append(
            prior_logpdf_at_initial_state
        )

    results_directory_name = "sampling_results"
    results_directory_path = working_directory / results_directory_name
    results_directory_path.mkdir(parents=True, exist_ok=False)

    tabular_results_file_base_name = (
        working_directory / "posterior_samples_table.out"
    )

    results_to_write = {}
    results_to_write["log_priors"] = list_of_prior_logpdf_at_initial_state
    results_to_write["log_likelihoods"] = (
        list_of_loglikelihood_at_initial_state
    )
    results_to_write["unnormalized_log_posteriors"] = (
        list_of_unnormalized_posterior_logpdf_at_initial_state
    )
    new_states_list = []
    for item in list_of_initial_states_of_model_parameters:
        if isinstance(item, list):
            new_states_list.append(item)
        else:
            new_states_list.append(item.tolist())
    results_to_write["new_states"] = new_states_list
    results_to_write["error_variances_scaled"] = (
        list_of_initial_states_of_error_variance_per_dataset
    )

    with open(results_directory_path / f"sample_0.json", "w") as f:
        json.dump(results_to_write, f, indent=4)

    adaptivity_results = {}
    # adaptivity_results["list_of_acceptance_rates"] = (
    #     list_of_acceptance_rates
    # )
    adaptivity_results["proposal_scale_list"] = proposal_scale_list
    cov_kernels_list = []
    for cov_kernel in list_of_proposal_covariance_kernels:
        cov_kernels_list.append(cov_kernel.tolist())
    adaptivity_results["list_of_proposal_covariance_kernels"] = (
        cov_kernels_list
    )
    with open(
        results_directory_path.parent / f"adaptivity_results_{0}.json", "w"
    ) as f:
        json.dump(adaptivity_results, f, indent=4)

    samples = metropolis_within_gibbs_sampler(
        uq_inputs,
        parallel_evaluation_function,
        function_to_evaluate,
        transformation_function,
        num_rv,
        num_edp,
        list_of_model_evaluation_functions,
        list_of_datasets,
        list_of_dataset_lengths,
        list_of_initial_states_of_model_parameters,
        list_of_cholesky_of_proposal_covariance_matrix,
        prior_inverse_gamma_parameters,
        loglikelihood_function,
        list_of_unnormalized_posterior_logpdf_at_initial_state,
        list_of_loglikelihood_at_initial_state,
        list_of_prior_logpdf_at_initial_state,
        prior_normal_inverse_wishart_parameters,
        results_directory_path,
        tabular_results_file_base_name,
        rv_inputs,
        edp_inputs,
        initial_state_of_hypermean,
        initial_state_of_hypercovariance,
        list_of_initial_states_of_error_variance_per_dataset,
        parent_distribution,
        num_accepts_list,
        proposal_scale_list,
        list_of_proposal_covariance_kernels,
    )


if __name__ == "__main__":
    input_args = sys.argv
    main(input_args)
