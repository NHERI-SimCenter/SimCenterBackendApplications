"""
authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, and Prof. J.P. Conte 
affiliation: University of California, San Diego
modified: Aakash Bangalore Satish, NHERI SimCenter, UC Berkeley
"""

import numpy as np
import tmcmcFunctions
import multiprocessing as mp
from multiprocessing import Pool
from runFEM import runFEM
from numpy.random import SeedSequence, default_rng
import os
import csv


def write_stage_start_info_to_logfile(logfile, stage_number, beta, effective_sample_size, 
                                      scale_factor_for_proposal_covariance, log_evidence, number_of_samples):
    logfile.write('\n\n\t\t==========================')
    logfile.write("\n\t\tStage number: {}".format(stage_number))
    if stage_number == 0:
        logfile.write("\n\t\tSampling from prior")
        logfile.write("\n\t\tbeta = 0")
    else:
        logfile.write("\n\t\tbeta = %9.8g" % beta)
    logfile.write("\n\t\tESS = %d" % effective_sample_size)
    logfile.write("\n\t\tscalem = %.2g" % scale_factor_for_proposal_covariance)
    logfile.write(f"\n\t\tlog-evidence = {log_evidence:<9.8g}")
    logfile.write("\n\n\t\tNumber of model evaluations in this stage: {}".format(number_of_samples))
    logfile.flush()
    os.fsync(logfile.fileno())

def write_eval_data_to_logfile(logfile, parallelize_MCMC, run_type, procCount=1, MPI_size=1, stage_num=0):
    if stage_num == 0:
        logfile.write("\n\n\t\tRun type: {}".format(run_type))
    if parallelize_MCMC:
        if run_type == "runningLocal":
            if stage_num == 0:
                logfile.write("\n\n\t\tCreated multiprocessing pool for runType: {}".format(run_type))
            else:
                logfile.write("\n\n\t\tLocal run - MCMC steps")
            logfile.write("\n\t\t\tNumber of processors being used: {}".format(procCount))
        else:
            if stage_num == 0:
                logfile.write("\n\n\t\tCreated mpi4py executor pool for runType: {}".format(run_type))
            else:
                logfile.write("\n\n\t\tRemote run - MCMC steps")
            logfile.write("\n\t\t\tmax_workers: {}".format(MPI_size))
    else:
        if stage_num == 0:
            logfile.write("\n\n\t\tNot parallelized")
        else:
            logfile.write("\n\n\t\tLocal run - MCMC steps, not parallelized")
        logfile.write("\n\t\t\tNumber of processors being used: {}".format(1))


def create_headings(logfile, model_number, model_parameters, edp_names_list, edp_lengths_list, writeOutputs):
    # Create the headings, which will be the first line of the file
    headings = 'eval_id\tinterface\t'
    if model_number == 0:
        logfile.write("\n\t\t\tCreating headings")
        for v in model_parameters['names']:
            headings += '{}\t'.format(v)
        if writeOutputs:  # create headings for outputs
            for i, edp in enumerate(edp_names_list):
                if edp_lengths_list[i] == 1:
                    headings += '{}\t'.format(edp)
                else:
                    for comp in range(edp_lengths_list[i]):
                        headings += '{}_{}\t'.format(edp, comp + 1)
        headings += '\n'
    
    return headings


def get_prediction_from_workdirs(i, working_directory):
    workdir_string = ("workdir." + str(i + 1))
    prediction = np.atleast_2d(np.genfromtxt(os.path.join(working_directory, workdir_string,
                                                            'results.out'))).reshape((1, -1))
    return prediction


def write_prior_data_to_files(logfile, working_directory, model_number, model_parameters, 
                              edp_names_list, edp_lengths_list, number_of_samples, dataToWrite, 
                              tab_file_name, predictions):
    
    tab_file_full_path = os.path.join(working_directory, tab_file_name)
    write_outputs = True
    headings = create_headings(logfile, model_number, model_parameters, edp_names_list, edp_lengths_list, write_outputs)

    logfile.write("\n\t\t\tWriting to file {}".format(tab_file_full_path))
    with open(tab_file_full_path, "a+") as f:
        if model_number == 0:
            f.write(headings)
        for i in range(number_of_samples):
            row_string = "{}\t{}\t".format(i + 1 + number_of_samples*model_number, model_number+1)
            for j in range(len(model_parameters['names'])):
                row_string += "{}\t".format(dataToWrite[i, j])
            if write_outputs:  # write the output data
                prediction = predictions[i, :]
                for pred in range(np.shape(prediction)[1]):
                    row_string += "{}\t".format(prediction[0, pred])
            row_string += "\n"
            f.write(row_string)

    logfile.write('\n\t\t==========================')
    logfile.flush()
    os.fsync(logfile.fileno())


def write_data_to_csvfile(logfile, total_number_of_models_in_ensemble, stage_number, model_number, 
                          working_directory, data_to_write):
    logfile.write("\n\n\t\tWriting samples from stage {} to csv file".format(stage_number - 1))
    if total_number_of_models_in_ensemble > 1:
        string_to_append = f'resultsStage{stage_number - 1}_Model_{model_number+1}.csv'
    else:
        string_to_append = f'resultsStage{stage_number - 1}.csv'
    resultsFilePath = os.path.join(os.path.abspath(working_directory), string_to_append)

    with open(resultsFilePath, 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerows(data_to_write)
    logfile.write("\n\t\t\tWrote to file {}".format(resultsFilePath))
    # Finished writing data


def RunTMCMC(number_of_samples, number_of_chains, all_distributions_list, number_of_MCMC_steps, max_number_of_MCMC_steps, 
             log_likelihood_function, model_parameters, working_directory, seed,
             calibration_data, number_of_experiments, covariance_matrix_list, edp_names_list, edp_lengths_list, scale_factors,
             shift_factors, run_type, logfile, MPI_size, driver_file, parallelize_MCMC=True, 
             model_number=0, total_number_of_models_in_ensemble=1):
    """ Runs TMCMC Algorithm """

    # Initialize (beta, effective sample size)
    beta = 0
    effective_sample_size = number_of_samples
    mytrace = []

    # Initialize other TMCMC variables
    number_of_MCMC_steps = number_of_MCMC_steps
    adaptively_calculate_num_MCMC_steps = True
    adaptively_scale_proposal_covariance = True  
    scale_factor_for_proposal_covariance = 1  # cov scale factor
    model_evidence = 1  # model evidence
    stage_number = 0  # stage number of TMCMC
    log_evidence = 0

    write_stage_start_info_to_logfile(logfile, stage_number, beta, effective_sample_size, 
                                      scale_factor_for_proposal_covariance, log_evidence, number_of_samples)
    # initial samples
    sample_values = tmcmcFunctions.initial_population(number_of_samples, all_distributions_list)

    # Evaluate posterior at Sm
    Priorm = np.array([tmcmcFunctions.log_prior(s, all_distributions_list) for s in sample_values]).squeeze()
    Postm = Priorm  # prior = post for beta = 0

    iterables = [(ind, sample_values[ind], model_parameters, working_directory, log_likelihood_function, calibration_data,
                  number_of_experiments, covariance_matrix_list, edp_names_list, edp_lengths_list,
                  scale_factors, shift_factors, driver_file) for ind in range(number_of_samples)]

    # Evaluate log-likelihood at current samples Sm
    if parallelize_MCMC:
        if run_type == "runningLocal":
            processor_count = mp.cpu_count()
            pool = Pool(processes=processor_count)
            write_eval_data_to_logfile(logfile, parallelize_MCMC, run_type, procCount=processor_count, stage_num=stage_number)
            Lmt = pool.starmap(runFEM, iterables)
        else:
            from mpi4py.futures import MPIPoolExecutor
            executor = MPIPoolExecutor(max_workers=MPI_size)
            write_eval_data_to_logfile(logfile, parallelize_MCMC, run_type, MPI_size=MPI_size, stage_num=stage_number)
            Lmt = list(executor.starmap(runFEM, iterables))
        Lm = np.array(Lmt).squeeze()
    else:
        write_eval_data_to_logfile(logfile, parallelize_MCMC, run_type, stage_num=stage_number)
        Lm = np.array([runFEM(ind, sample_values[ind], model_parameters, working_directory, log_likelihood_function,
                              calibration_data, number_of_experiments, covariance_matrix_list,
                              edp_names_list, edp_lengths_list, scale_factors,
                              shift_factors, driver_file)
                       for ind in range(number_of_samples)]).squeeze()

    total_number_of_model_evaluations = number_of_samples
    logfile.write("\n\n\t\tTotal number of model evaluations so far: {}".format(total_number_of_model_evaluations))

    # Write the results of the first stage to a file named dakotaTabPrior.out for quoFEM to be able to read the results
    logfile.write("\n\n\t\tWriting prior samples to 'dakotaTabPrior.out' for quoFEM to read the results")
    predictions = []
    for i in range(number_of_samples):
        predictions.append(get_prediction_from_workdirs(i, working_directory))
    predictions = np.atleast_2d(np.array(predictions))
    write_prior_data_to_files(logfile, working_directory, model_number, model_parameters, 
                              edp_names_list, edp_lengths_list, number_of_samples, dataToWrite=sample_values, 
                              tab_file_name="dakotaTabPrior.out", predictions=predictions)

    total_log_evidence = 0

    while beta < 1:
        stage_number += 1
        # adaptively compute beta s.t. ESS = N/2 or ESS = 0.95*prev_ESS
        # plausible weights of Sm corresponding to new beta
        # beta, Wm, ESS = tmcmcFunctions.compute_beta(beta, Lm, ESS, threshold=0.95)
        # beta, Wm, ESS = tmcmcFunctions.compute_beta(beta, Lm, ESS, threshold=0.5)
        beta, log_evidence, Wm, effective_sample_size = tmcmcFunctions.compute_beta_evidence(beta, Lm, logfile, threshold=1.0)

        total_log_evidence = total_log_evidence + log_evidence

        # seed to reproduce results
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(number_of_samples + 1)

        # update model evidence
        model_evidence = model_evidence * (sum(Wm) / number_of_samples)

        # Calculate covariance matrix using Wm_n
        Cm = np.cov(sample_values, aweights=Wm / sum(Wm), rowvar=False)
        # logFile.write("\nCovariance matrix: {}".format(Cm))

        # Resample ###################################################
        # Resampling using plausible weights
        # SmcapIDs = np.random.choice(range(N), N, p=Wm / sum(Wm))
        rng = default_rng(child_seeds[-1])
        SmcapIDs = rng.choice(range(number_of_samples), number_of_samples, p=Wm/sum(Wm))

        Smcap = sample_values[SmcapIDs]
        Lmcap = Lm[SmcapIDs]
        Postmcap = Postm[SmcapIDs]

        # save to trace
        # stage m: samples, likelihood, weights, next stage ESS, next stage beta, resampled samples
        mytrace.append([sample_values, Lm, Wm, effective_sample_size, beta, Smcap])

        # Write Data to '.csv' files
        data_to_write = mytrace[stage_number - 1][0]
        write_data_to_csvfile(logfile, total_number_of_models_in_ensemble, stage_number, model_number, 
                                working_directory, data_to_write)

        # Perturb ###################################################
        # perform MCMC starting at each Smcap (total: N) for Nm_steps
        Em = (scale_factor_for_proposal_covariance ** 2) * Cm  # Proposal dist covariance matrix

        numProposals = number_of_chains * number_of_MCMC_steps
        total_number_of_model_evaluations += numProposals

        write_stage_start_info_to_logfile(logfile, stage_number, beta, effective_sample_size, 
                                      scale_factor_for_proposal_covariance, log_evidence, numProposals)

        numAccepts = 0
        iterables = [(j1, Em, number_of_MCMC_steps, Smcap[j1], Lmcap[j1], Postmcap[j1], beta, 
                      numAccepts, all_distributions_list, log_likelihood_function, model_parameters,
                      working_directory, default_rng(child_seeds[j1]),
                      calibration_data, number_of_experiments, covariance_matrix_list,
                      edp_names_list, edp_lengths_list, scale_factors,
                      shift_factors, driver_file)
                      for j1 in range(number_of_samples)]
        
        if parallelize_MCMC:
            if run_type == "runningLocal":
                write_eval_data_to_logfile(logfile, parallelize_MCMC, run_type, procCount=processor_count, stage_num=stage_number)
                results = pool.starmap(tmcmcFunctions.MCMC_MH, iterables)
            else:
                write_eval_data_to_logfile(logfile, parallelize_MCMC, run_type, MPI_size=MPI_size, stage_num=stage_number)
                results = list(executor.starmap(tmcmcFunctions.MCMC_MH, iterables))
        else:
            write_eval_data_to_logfile(logfile, parallelize_MCMC, run_type, stage_num=stage_number)
            results = [tmcmcFunctions.MCMC_MH(j1, Em, number_of_MCMC_steps, Smcap[j1], Lmcap[j1], 
                                              Postmcap[j1], beta, numAccepts, all_distributions_list,
                                              log_likelihood_function, model_parameters, working_directory, 
                                              default_rng(child_seeds[j1]),
                                              calibration_data, number_of_experiments, covariance_matrix_list,
                                              edp_names_list, edp_lengths_list, scale_factors, shift_factors, driver_file)
                        for j1 in range(number_of_samples)]

        Sm1, Lm1, Postm1, numAcceptsS, all_proposals, all_PLP = zip(*results)
        Sm1 = np.asarray(Sm1)
        Lm1 = np.asarray(Lm1)
        Postm1 = np.asarray(Postm1)
        numAcceptsS = np.asarray(numAcceptsS)
        numAccepts = sum(numAcceptsS)
        all_proposals = np.asarray(all_proposals)
        all_PLP = np.asarray(all_PLP)

        logfile.write("\n\n\t\tTotal number of model evaluations so far: {}".format(total_number_of_model_evaluations))

        # total observed acceptance rate
        R = numAccepts / numProposals
        if R < 1e-5:
            logfile.write("\n\n\t\tacceptance rate = %9.5g" % R)
        else:
            logfile.write("\n\n\t\tacceptance rate = %.6g" % R)

        # Calculate Nm_steps based on observed acceptance rate
        if adaptively_calculate_num_MCMC_steps:
            # increase max Nmcmc with stage number
            number_of_MCMC_steps = min(number_of_MCMC_steps + 1, max_number_of_MCMC_steps)
            logfile.write("\n\t\tadapted max MCMC steps = %d" % number_of_MCMC_steps)

            acc_rate = max(1. / numProposals, R)
            number_of_MCMC_steps = min(number_of_MCMC_steps, 1 + int(np.log(1 - 0.99) / np.log(1 - acc_rate)))
            logfile.write("\n\t\tnext MCMC Nsteps = %d" % number_of_MCMC_steps)

        logfile.write('\n\t\t==========================')

        # scale factor based on observed acceptance ratio
        if adaptively_scale_proposal_covariance:
            scale_factor_for_proposal_covariance = (1 / 9) + ((8 / 9) * R)

        # for next beta
        sample_values, Postm, Lm = Sm1, Postm1, Lm1

    # save to trace
    mytrace.append([sample_values, Lm, np.ones(len(Wm)), 'notValid', 1, 'notValid'])

    # Write last stage data to '.csv' file
    data_to_write = mytrace[stage_number][0]
    logfile.write("\n\n\t\tWriting samples from stage {} to csv file".format(stage_number))

    if total_number_of_models_in_ensemble > 1:
        stringToAppend = f'resultsStage{stage_number}_Model_{model_number+1}.csv'
    else:
        stringToAppend = f'resultsStage{stage_number}.csv'
    resultsFilePath = os.path.join(os.path.abspath(working_directory), stringToAppend)

    with open(resultsFilePath, 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerows(data_to_write)
    logfile.write("\n\t\t\tWrote to file {}".format(resultsFilePath))

    if parallelize_MCMC == 'yes':
        if run_type == "runningLocal":
            pool.close()
            logfile.write("\n\tClosed multiprocessing pool for runType: {}".format(run_type))
        else:
            executor.shutdown()
            logfile.write("\n\tShutdown mpi4py executor pool for runType: {}".format(run_type))

    return mytrace, total_log_evidence
