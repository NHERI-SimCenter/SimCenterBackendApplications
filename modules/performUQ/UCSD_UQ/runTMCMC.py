"""authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, and Prof. J.P. Conte
affiliation: University of California, San Diego
modified: Aakash Bangalore Satish, NHERI SimCenter, UC Berkeley
"""  # noqa: INP001, D205, D400

import csv
import multiprocessing as mp
import os
from multiprocessing import Pool

import numpy as np
import tmcmcFunctions
from numpy.random import SeedSequence, default_rng
from runFEM import runFEM


def write_stage_start_info_to_logfile(  # noqa: D103
    logfile,
    stage_number,
    beta,
    effective_sample_size,
    scale_factor_for_proposal_covariance,
    log_evidence,
    number_of_samples,
):
    logfile.write('\n\n\t\t==========================')
    logfile.write(f'\n\t\tStage number: {stage_number}')
    if stage_number == 0:
        logfile.write('\n\t\tSampling from prior')
        logfile.write('\n\t\tbeta = 0')
    else:
        logfile.write('\n\t\tbeta = %9.8g' % beta)  # noqa: UP031
    logfile.write('\n\t\tESS = %d' % effective_sample_size)
    logfile.write('\n\t\tscalem = %.2g' % scale_factor_for_proposal_covariance)  # noqa: UP031
    logfile.write(f'\n\t\tlog-evidence = {log_evidence:<9.8g}')
    logfile.write(
        f'\n\n\t\tNumber of model evaluations in this stage: {number_of_samples}'
    )
    logfile.flush()
    os.fsync(logfile.fileno())


def write_eval_data_to_logfile(  # noqa: D103
    logfile,
    parallelize_MCMC,  # noqa: N803
    run_type,
    proc_count=1,
    MPI_size=1,  # noqa: N803
    stage_num=0,
):
    if stage_num == 0:
        logfile.write(f'\n\n\t\tRun type: {run_type}')
    if parallelize_MCMC:
        if run_type == 'runningLocal':
            if stage_num == 0:
                logfile.write(
                    f'\n\n\t\tCreated multiprocessing pool for runType: {run_type}'
                )
            else:
                logfile.write('\n\n\t\tLocal run - MCMC steps')
            logfile.write(f'\n\t\t\tNumber of processors being used: {proc_count}')
        else:
            if stage_num == 0:
                logfile.write(
                    f'\n\n\t\tCreated mpi4py executor pool for runType: {run_type}'
                )
            else:
                logfile.write('\n\n\t\tRemote run - MCMC steps')
            logfile.write(f'\n\t\t\tmax_workers: {MPI_size}')
    else:
        if stage_num == 0:
            logfile.write('\n\n\t\tNot parallelized')
        else:
            logfile.write('\n\n\t\tLocal run - MCMC steps, not parallelized')
        logfile.write(f'\n\t\t\tNumber of processors being used: {1}')


def create_headings(  # noqa: D103
    logfile,
    model_number,
    model_parameters,
    edp_names_list,
    edp_lengths_list,
    writeOutputs,  # noqa: N803
):
    # Create the headings, which will be the first line of the file
    headings = 'eval_id\tinterface\t'
    if model_number == 0:
        logfile.write('\n\t\t\tCreating headings')
        for v in model_parameters['names']:
            headings += f'{v}\t'
        if writeOutputs:  # create headings for outputs
            for i, edp in enumerate(edp_names_list):
                if edp_lengths_list[i] == 1:
                    headings += f'{edp}\t'
                else:
                    for comp in range(edp_lengths_list[i]):
                        headings += f'{edp}_{comp + 1}\t'
        headings += '\n'

    return headings


def get_prediction_from_workdirs(i, working_directory):  # noqa: D103
    workdir_string = 'workdir.' + str(i + 1)
    prediction = np.atleast_2d(
        np.genfromtxt(os.path.join(working_directory, workdir_string, 'results.out'))  # noqa: PTH118
    ).reshape((1, -1))
    return prediction  # noqa: RET504


def write_data_to_tab_files(  # noqa: D103
    logfile,
    working_directory,
    model_number,
    model_parameters,
    edp_names_list,
    edp_lengths_list,
    number_of_samples,
    dataToWrite,  # noqa: N803
    tab_file_name,
    predictions,
):
    tab_file_full_path = os.path.join(working_directory, tab_file_name)  # noqa: PTH118
    write_outputs = True
    headings = create_headings(
        logfile,
        model_number,
        model_parameters,
        edp_names_list,
        edp_lengths_list,
        write_outputs,
    )

    logfile.write(f'\n\t\t\tWriting to file {tab_file_full_path}')
    with open(tab_file_full_path, 'a+') as f:  # noqa: PTH123
        if model_number == 0:
            f.write(headings)
        for i in range(number_of_samples):
            row_string = (
                f'{i + 1 + number_of_samples * model_number}\t{model_number + 1}\t'
            )
            for j in range(len(model_parameters['names'])):
                row_string += f'{dataToWrite[i, j]}\t'
            if write_outputs:  # write the output data
                prediction = predictions[i, :]
                for pred in prediction:
                    row_string += f'{pred}\t'
            row_string += '\n'
            f.write(row_string)

    logfile.write('\n\t\t==========================')
    logfile.flush()
    os.fsync(logfile.fileno())


def write_data_to_csvfile(  # noqa: D103
    logfile,
    total_number_of_models_in_ensemble,
    stage_number,
    model_number,
    working_directory,
    data_to_write,
):
    logfile.write(
        f'\n\n\t\tWriting samples from stage {stage_number - 1} to csv file'
    )
    if total_number_of_models_in_ensemble > 1:
        string_to_append = (
            f'resultsStage{stage_number - 1}_Model_{model_number + 1}.csv'
        )
    else:
        string_to_append = f'resultsStage{stage_number - 1}.csv'
    resultsFilePath = os.path.join(  # noqa: PTH118, N806
        os.path.abspath(working_directory),  # noqa: PTH100
        string_to_append,
    )

    with open(resultsFilePath, 'w', newline='') as csvfile:  # noqa: PTH123
        csvWriter = csv.writer(csvfile)  # noqa: N806
        csvWriter.writerows(data_to_write)
    logfile.write(f'\n\t\t\tWrote to file {resultsFilePath}')
    # Finished writing data


def run_TMCMC(  # noqa: N802, PLR0913
    number_of_samples,
    number_of_chains,
    all_distributions_list,
    number_of_MCMC_steps,  # noqa: N803
    max_number_of_MCMC_steps,  # noqa: N803
    log_likelihood_function,
    model_parameters,
    working_directory,
    seed,
    calibration_data,
    number_of_experiments,
    covariance_matrix_list,
    edp_names_list,
    edp_lengths_list,
    scale_factors,
    shift_factors,
    run_type,
    logfile,
    MPI_size,  # noqa: N803
    driver_file,
    parallelize_MCMC=True,  # noqa: FBT002, N803
    model_number=0,
    total_number_of_models_in_ensemble=1,
):
    """Runs TMCMC Algorithm"""  # noqa: D400, D401
    # Initialize (beta, effective sample size)
    beta = 0
    effective_sample_size = number_of_samples
    mytrace = []

    # Initialize other TMCMC variables
    number_of_MCMC_steps = number_of_MCMC_steps  # noqa: N806, PLW0127
    adaptively_calculate_num_MCMC_steps = True  # noqa: N806
    adaptively_scale_proposal_covariance = True
    scale_factor_for_proposal_covariance = 1  # cov scale factor
    # model_evidence = 1  # model evidence
    stage_number = 0  # stage number of TMCMC
    log_evidence = 0

    write_stage_start_info_to_logfile(
        logfile,
        stage_number,
        beta,
        effective_sample_size,
        scale_factor_for_proposal_covariance,
        log_evidence,
        number_of_samples,
    )
    # initial samples
    sample_values = tmcmcFunctions.initial_population(
        number_of_samples, all_distributions_list
    )

    # Evaluate posterior at Sm
    prior_pdf_values = np.array(
        [tmcmcFunctions.log_prior(s, all_distributions_list) for s in sample_values]
    ).squeeze()
    unnormalized_posterior_pdf_values = prior_pdf_values  # prior = post for beta = 0

    iterables = [
        (
            ind,
            sample_values[ind],
            model_parameters,
            working_directory,
            log_likelihood_function,
            calibration_data,
            number_of_experiments,
            covariance_matrix_list,
            edp_names_list,
            edp_lengths_list,
            scale_factors,
            shift_factors,
            driver_file,
        )
        for ind in range(number_of_samples)
    ]

    # Evaluate log-likelihood at current samples Sm
    if run_type == 'runningLocal':
        processor_count = mp.cpu_count()
        pool = Pool(processes=processor_count)
        write_eval_data_to_logfile(
            logfile,
            parallelize_MCMC,
            run_type,
            proc_count=processor_count,
            stage_num=stage_number,
        )
        outputs = pool.starmap(runFEM, iterables)
        log_likelihoods_list = []
        predictions_list = []
        for output in outputs:
            log_likelihoods_list.append(output[0])
            predictions_list.append(output[1])
    else:
        from mpi4py.futures import MPIPoolExecutor

        executor = MPIPoolExecutor(max_workers=MPI_size)
        write_eval_data_to_logfile(
            logfile,
            parallelize_MCMC,
            run_type,
            MPI_size=MPI_size,
            stage_num=stage_number,
        )
        outputs = list(executor.starmap(runFEM, iterables))
        log_likelihoods_list = []
        predictions_list = []
        for output in outputs:
            log_likelihoods_list.append(output[0])
            predictions_list.append(output[1])
    log_likelihood_values = np.array(log_likelihoods_list).squeeze()
    prediction_values = np.array(predictions_list).reshape((number_of_samples, -1))

    total_number_of_model_evaluations = number_of_samples
    logfile.write(
        f'\n\n\t\tTotal number of model evaluations so far: {total_number_of_model_evaluations}'
    )

    # Write the results of the first stage to a file named dakotaTabPrior.out for quoFEM to be able to read the results
    logfile.write(
        "\n\n\t\tWriting prior samples to 'dakotaTabPrior.out' for quoFEM to read the results"
    )
    write_data_to_tab_files(
        logfile,
        working_directory,
        model_number,
        model_parameters,
        edp_names_list,
        edp_lengths_list,
        number_of_samples,
        dataToWrite=sample_values,
        tab_file_name='dakotaTabPrior.out',
        predictions=prediction_values,
    )

    total_log_evidence = 0

    while beta < 1:
        stage_number += 1
        # adaptively compute beta s.t. ESS = N/2 or ESS = 0.95*prev_ESS
        # plausible weights of Sm corresponding to new beta
        # beta, Wm, ESS = tmcmcFunctions.compute_beta(beta, Lm, ESS, threshold=0.95)
        # beta, Wm, ESS = tmcmcFunctions.compute_beta(beta, Lm, ESS, threshold=0.5)
        beta, log_evidence, weights, effective_sample_size = (
            tmcmcFunctions.compute_beta_evidence(
                beta, log_likelihood_values, logfile, threshold=1.0
            )
        )
        # beta, log_evidence, weights, effective_sample_size = tmcmcFunctions.compute_beta_evidence_old(beta, log_likelihood_values, logfile, int(effective_sample_size/2), threshold=1.0)

        total_log_evidence = total_log_evidence + log_evidence

        # seed to reproduce results
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(number_of_samples + 1)

        # update model evidence
        # model_evidence = model_evidence * (sum(weights) / number_of_samples)

        # Calculate covariance matrix using Wm_n
        weighted_sample_covariance_matrix = np.cov(
            sample_values, aweights=weights, rowvar=False
        )
        # logFile.write("\nCovariance matrix: {}".format(Cm))

        # Resample ###################################################
        # Resampling using plausible weights
        # SmcapIDs = np.random.choice(range(N), N, p=Wm / sum(Wm))
        rng = default_rng(child_seeds[-1])
        resample_ids = rng.choice(
            range(number_of_samples), number_of_samples, p=weights
        )

        resampled_values = sample_values[resample_ids]
        resampled_log_likelihood_values = log_likelihood_values[resample_ids]
        resampled_unnormalized_posterior_pdf_values = (
            unnormalized_posterior_pdf_values[resample_ids]
        )
        resampled_prediction_values = np.atleast_2d(
            prediction_values[resample_ids, :]
        )

        # save to trace
        # stage m: samples, likelihood, weights, next stage ESS, next stage beta, resampled samples
        mytrace.append(
            [
                sample_values,
                log_likelihood_values,
                weights,
                effective_sample_size,
                beta,
                resampled_values,
            ]
        )

        # Write Data to '.csv' files
        data_to_write = np.hstack((sample_values, prediction_values))
        write_data_to_csvfile(
            logfile,
            total_number_of_models_in_ensemble,
            stage_number,
            model_number,
            working_directory,
            data_to_write,
        )

        # Perturb ###################################################
        # perform MCMC starting at each Smcap (total: N) for Nm_steps
        scaled_proposal_covariance_matrix = (
            scale_factor_for_proposal_covariance**2
        ) * weighted_sample_covariance_matrix  # Proposal dist covariance matrix

        number_of_model_evaluations_in_this_stage = (
            number_of_chains * number_of_MCMC_steps
        )
        write_stage_start_info_to_logfile(
            logfile,
            stage_number,
            beta,
            effective_sample_size,
            scale_factor_for_proposal_covariance,
            log_evidence,
            number_of_model_evaluations_in_this_stage,
        )

        number_of_accepted_states_in_this_stage = 0
        iterables = [
            (
                sample_num,
                scaled_proposal_covariance_matrix,
                number_of_MCMC_steps,
                resampled_values[sample_num],
                resampled_log_likelihood_values[sample_num],
                resampled_unnormalized_posterior_pdf_values[sample_num],
                beta,
                number_of_accepted_states_in_this_stage,
                all_distributions_list,
                log_likelihood_function,
                model_parameters,
                working_directory,
                default_rng(child_seeds[sample_num]),
                calibration_data,
                number_of_experiments,
                covariance_matrix_list,
                edp_names_list,
                edp_lengths_list,
                scale_factors,
                shift_factors,
                driver_file,
                resampled_prediction_values[sample_num, :].reshape((1, -1)),
            )
            for sample_num in range(number_of_samples)
        ]

        if run_type == 'runningLocal':
            write_eval_data_to_logfile(
                logfile,
                parallelize_MCMC,
                run_type,
                proc_count=processor_count,
                stage_num=stage_number,
            )
            results = pool.starmap(tmcmcFunctions.MCMC_MH, iterables)
        else:
            write_eval_data_to_logfile(
                logfile,
                parallelize_MCMC,
                run_type,
                MPI_size=MPI_size,
                stage_num=stage_number,
            )
            results = list(executor.starmap(tmcmcFunctions.MCMC_MH, iterables))

        (
            samples_list,
            loglikes_list,
            posterior_pdf_vals_list,
            num_accepts,
            all_proposals,
            all_PLP,  # noqa: N806
            preds_list,
        ) = zip(*results)
        # for next beta
        sample_values = np.asarray(samples_list)
        log_likelihood_values = np.asarray(loglikes_list)
        unnormalized_posterior_pdf_values = np.asarray(posterior_pdf_vals_list)
        prediction_values = np.asarray(preds_list).reshape((number_of_samples, -1))

        num_accepts = np.asarray(num_accepts)
        number_of_accepted_states_in_this_stage = sum(num_accepts)
        all_proposals = np.asarray(all_proposals)
        all_PLP = np.asarray(all_PLP)  # noqa: N806

        total_number_of_model_evaluations += (
            number_of_model_evaluations_in_this_stage
        )
        logfile.write(
            f'\n\n\t\tTotal number of model evaluations so far: {total_number_of_model_evaluations}'
        )

        # total observed acceptance rate
        R = (  # noqa: N806
            number_of_accepted_states_in_this_stage
            / number_of_model_evaluations_in_this_stage
        )
        logfile.write(f'\n\n\t\tacceptance rate = {R:<9.6g}')
        if (
            adaptively_scale_proposal_covariance
        ):  # scale factor based on observed acceptance ratio
            scale_factor_for_proposal_covariance = (1 / 9) + ((8 / 9) * R)

        if (
            adaptively_calculate_num_MCMC_steps
        ):  # Calculate Nm_steps based on observed acceptance rate
            # increase max Nmcmc with stage number
            number_of_MCMC_steps = min(  # noqa: N806
                number_of_MCMC_steps + 1, max_number_of_MCMC_steps
            )
            logfile.write('\n\t\tadapted max MCMC steps = %d' % number_of_MCMC_steps)

            acc_rate = max(1.0 / number_of_model_evaluations_in_this_stage, R)
            number_of_MCMC_steps = min(  # noqa: N806
                number_of_MCMC_steps,
                1 + int(np.log(1 - 0.99) / np.log(1 - acc_rate)),
            )
            logfile.write('\n\t\tnext MCMC Nsteps = %d' % number_of_MCMC_steps)

        logfile.write('\n\t\t==========================')

    # save to trace
    mytrace.append(
        [
            sample_values,
            log_likelihood_values,
            np.ones(len(weights)),
            'notValid',
            1,
            'notValid',
        ]
    )

    # Write last stage data to '.csv' file
    data_to_write = np.hstack((sample_values, prediction_values))
    write_data_to_csvfile(
        logfile,
        total_number_of_models_in_ensemble,
        stage_number,
        model_number,
        working_directory,
        data_to_write,
    )

    write_data_to_tab_files(
        logfile,
        working_directory,
        model_number,
        model_parameters,
        edp_names_list,
        edp_lengths_list,
        number_of_samples,
        dataToWrite=sample_values,
        tab_file_name='dakotaTab.out',
        predictions=prediction_values,
    )

    if parallelize_MCMC == 'yes':
        if run_type == 'runningLocal':
            pool.close()
            logfile.write(f'\n\tClosed multiprocessing pool for runType: {run_type}')
        else:
            executor.shutdown()
            logfile.write(
                f'\n\tShutdown mpi4py executor pool for runType: {run_type}'
            )

    return mytrace, total_log_evidence  # noqa: DOC201
