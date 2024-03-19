import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.stats

import preprocess_hierarchical_bayesian

path_to_common_uq = Path(__file__).parent.parent / "common"
sys.path.append(str(path_to_common_uq))
import uq_utilities
import mwg_sampler


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
    working_directory = Path(input_args[0]).resolve()
    template_directory = Path(input_args[1]).resolve()
    run_type = input_args[2]  # either "runningLocal" or "runningRemote"
    workflow_driver = input_args[3]
    input_file = input_args[4]

    # input_file_full_path = template_directory / input_file

    with open(input_file, "r") as f:
        inputs = json.load(f)

    uq_inputs = inputs["UQ"]
    rv_inputs = inputs["randomVariables"]
    edp_inputs = inputs["EDP"]

    (
        parallel_pool,
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

    prior_inverse_gamma_parameters = uq_utilities.InverseGammaParameters(
        alpha_scalar=1 / 2, beta_scalar=1 / 2
    )

    prior_normal_inverse_wishart_parameters = (
        uq_utilities.NormalInverseWishartParameters(
            mu_vector=np.zeros((num_rv, 1)),
            lambda_scalar=0,
            nu_scalar=num_rv + 2,
            psi_matrix=np.eye(num_rv),
        )
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

    parallel_evaluation_function = parallel_pool.pool.starmap

    list_of_model_outputs = list(
        parallel_evaluation_function(function_to_evaluate, iterable)
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

    samples = mwg_sampler.metropolis_within_gibbs_sampler(
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

    if run_type == "runningRemote":
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(0)


if __name__ == "__main__":
    input_args = sys.argv
    main(input_args)
