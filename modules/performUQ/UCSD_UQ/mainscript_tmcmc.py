"""authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, Prof. J.P. Conte, Aakash Bangalore Satish*
affiliation: University of California, San Diego, *SimCenter, University of California, Berkeley

"""  # noqa: CPY001, D205, D400, INP001

# ======================================================================================================================
import os
import sys
import time
from typing import TextIO

import numpy as np
from calibration_utilities import (
    CalDataPreparer,
    CovarianceMatrixPreparer,
    DataTransformer,
    LogLikelihoodHandler,
    createLogFile,
    make_distributions,
    syncLogFile,
)
from parseData import parseDataFunction
from runTMCMC import run_TMCMC

# ======================================================================================================================


def computeModelPosteriorProbabilities(modelPriorProbabilities, modelEvidences):  # noqa: N802, N803, D103
    denominator = np.dot(modelPriorProbabilities, modelEvidences)
    return modelPriorProbabilities * modelEvidences / denominator


def computeModelPosteriorProbabilitiesUsingLogEvidences(  # noqa: N802, D103
    modelPriorProbabilities,  # noqa: N803
    modelLogEvidences,  # noqa: N803
):
    deltas = modelLogEvidences - np.min(modelLogEvidences)
    denominator = np.dot(modelPriorProbabilities, np.exp(deltas))
    return modelPriorProbabilities * np.exp(deltas) / denominator


# ======================================================================================================================


class TMCMC_Data:  # noqa: D101
    def __init__(
        self,
        mainscriptPath: str,  # noqa: N803
        workdirMain: str,  # noqa: N803
        runType: str,  # noqa: N803
        workflowDriver: str,  # noqa: N803
        logFile: TextIO,  # noqa: N803
        numBurnInSteps: int = 10,  # noqa: N803
    ) -> None:
        self.mainscriptPath = mainscriptPath
        self.workdirMain = workdirMain
        self.runType = runType
        self.workflowDriver = workflowDriver
        self.logFile = logFile

        self.MPI_size = self.getMPI_size()
        self.parallelizeMCMC = True

        self.recommendedNumChains = 50
        self.numBurnInSteps = numBurnInSteps
        self.numSkipSteps = 1

    def getMPI_size(self):  # noqa: N802, D102
        if self.runType == 'runningRemote':
            from mpi4py import MPI  # noqa: PLC0415

            self.comm = MPI.COMM_WORLD
            self.MPI_size = self.comm.Get_size()

    def updateUQInfo(self, numberOfSamples, seedVal):  # noqa: N802, N803, D102
        self.numberOfSamples = numberOfSamples
        self.seedVal = seedVal

    def findNumProcessorsAvailable(self):  # noqa: N802, D102
        if self.runType == 'runningLocal':
            import multiprocessing as mp  # noqa: PLC0415

            self.numProcessors = mp.cpu_count()
        elif self.runType == 'runningRemote':
            from mpi4py import MPI  # noqa: PLC0415

            self.comm = MPI.COMM_WORLD
            self.numProcessors = self.comm.Get_size()
        else:
            self.numProcessors = 1

    def getNumChains(self, numberOfSamples, runType, numProcessors):  # noqa: N802, N803, D102
        if runType == 'runningLocal':
            self.numChains = int(min(numProcessors, self.recommendedNumChains))
        elif runType == 'runningRemote':
            self.numChains = int(max(numProcessors, self.recommendedNumChains))
        else:
            self.numChains = self.recommendedNumChains

        self.numChains = max(self.numChains, numberOfSamples)

    def getNumStepsPerChainAfterBurnIn(self, numParticles, numChains):  # noqa: N802, N803, D102
        self.numStepsAfterBurnIn = (
            int(np.ceil(numParticles / numChains)) * self.numSkipSteps
        )
        # self.numStepsPerChain = numBurnInSteps + numStepsAfterBurnIn


# ======================================================================================================================


# ======================================================================================================================
def main(input_args):  # noqa: D103
    t1 = time.time()

    # Initialize analysis
    # mainscript_path = os.path.abspath(input_args[0])
    # working_directory = os.path.abspath(input_args[1])
    # template_directory = os.path.abspath(input_args[2])
    # run_type = input_args[3]  # either "runningLocal" or "runningRemote"
    # driver_file = input_args[4]
    # input_json_filename = input_args[5]

    mainscript_path = os.path.abspath(__file__)  # noqa: PTH100
    working_directory = os.path.abspath(input_args[0])  # noqa: PTH100
    template_directory = os.path.abspath(input_args[1])  # noqa: PTH100
    run_type = input_args[2]  # either "runningLocal" or "runningRemote"
    driver_file = input_args[3]
    input_json_filename = input_args[4]

    logfile_name = 'logFileTMCMC.txt'
    logfile = createLogFile(where=working_directory, logfile_name=logfile_name)

    # Remove dakotaTab and dakotaTabPrior files if they already exist in the working directory
    try:
        os.remove('dakotaTab.out')  # noqa: PTH107
        os.remove('dakotTabPrior.out')  # noqa: PTH107
    except OSError:
        pass

    # # ================================================================================================================

    # Process input json file
    # input_json_filename_full_path = os.path.join(os.path.abspath(template_directory), input_json_filename)
    input_json_filename_full_path = input_json_filename
    logfile.write('\n\n==========================')
    logfile.write(f'\nParsing the json input file {input_json_filename_full_path}')
    (
        number_of_samples,
        seed_value,
        calibration_data_filename,
        loglikelihood_module,  # noqa: F841
        write_outputs,  # noqa: F841
        variables_list,
        edp_names_list,
        edp_lengths_list,
        models_dict,  # noqa: F841
        total_number_of_models_in_ensemble,
    ) = parseDataFunction(
        input_json_filename_full_path,
        logfile,
        working_directory,
        os.path.dirname(mainscript_path),  # noqa: PTH120
    )
    syncLogFile(logfile)

    # # ================================================================================================================

    # Initialize TMCMC object
    tmcmc_data_instance = TMCMC_Data(
        mainscript_path,
        working_directory,
        run_type,
        driver_file,
        logfile,
        numBurnInSteps=4,
    )
    tmcmc_data_instance.updateUQInfo(number_of_samples, seed_value)
    tmcmc_data_instance.findNumProcessorsAvailable()
    tmcmc_data_instance.getNumChains(
        number_of_samples, run_type, tmcmc_data_instance.numProcessors
    )
    tmcmc_data_instance.getNumStepsPerChainAfterBurnIn(
        number_of_samples, tmcmc_data_instance.numChains
    )

    # # ================================================================================================================

    # Read calibration data
    data_preparer_instance = CalDataPreparer(
        working_directory,
        template_directory,
        calibration_data_filename,
        edp_names_list,
        edp_lengths_list,
        logfile,
    )
    calibration_data, number_of_experiments = (
        data_preparer_instance.getCalibrationData()
    )

    # # ================================================================================================================

    # Transform the data depending on the option chosen by the user
    transformation = 'absMaxScaling'
    data_transformer_instance = DataTransformer(
        transformStrategy=transformation, logFile=logfile
    )

    scale_factors, shift_factors = (
        data_transformer_instance.computeScaleAndShiftFactors(
            calibration_data, edp_lengths_list
        )
    )
    logfile.write('\n\n\tThe scale and shift factors computed are: ')
    for j in range(len(edp_names_list)):
        logfile.write(
            f'\n\t\tEDP: {edp_names_list[j]}, scale factor: {scale_factors[j]}, shift factor: {shift_factors[j]}'
        )

    transformed_calibration_data = data_transformer_instance.transformData()
    logfile.write(
        f'\n\nThe transformed calibration data: \n{transformed_calibration_data}'
    )

    # ======================================================================================================================
    # Process covariance matrix options
    cov_matrix_options_instance = CovarianceMatrixPreparer(
        transformed_calibration_data,
        edp_lengths_list,
        edp_names_list,
        working_directory,
        number_of_experiments,
        logfile,
        run_type,
    )
    defaultErrorVariances = cov_matrix_options_instance.getDefaultErrorVariances()  # noqa: N806, F841
    covariance_matrix_list = cov_matrix_options_instance.createCovarianceMatrix()

    # ======================================================================================================================
    # Get log-likelihood function
    LL_Handler = LogLikelihoodHandler(  # noqa: N806
        data=transformed_calibration_data,
        covariance_matrix_blocks_list=covariance_matrix_list,
        list_of_data_segment_lengths=edp_lengths_list,
        list_of_scale_factors=scale_factors,
        list_of_shift_factors=shift_factors,
        workdir_main=working_directory,
        full_path_to_tmcmc_code_directory=mainscript_path,
        log_likelihood_file_name='loglike_script.py',
    )
    log_likelihood_function = LL_Handler.evaluate_log_likelihood

    # ======================================================================================================================
    # Start TMCMC workflow
    logfile.write('\n\n==========================')
    logfile.write('\nSetting up the TMCMC algorithm')

    # sys.path.append(workdirMain)
    logfile.write(f'\n\tResults path: {working_directory}')

    # number of particles: Np
    number_of_samples = tmcmc_data_instance.numberOfSamples
    logfile.write(f'\n\tNumber of particles: {number_of_samples}')

    # number of max MCMC steps
    number_of_MCMC_steps = (  # noqa: N806
        tmcmc_data_instance.numBurnInSteps + tmcmc_data_instance.numStepsAfterBurnIn
    )
    max_number_of_MCMC_steps = 10  # noqa: N806
    logfile.write(f'\n\tNumber of MCMC steps in first stage: {number_of_MCMC_steps}')
    logfile.write(
        f'\n\tMax. number of MCMC steps in any stage: {max_number_of_MCMC_steps}'
    )

    syncLogFile(logfile)

    # ======================================================================================================================
    # Initialize variables to store prior model probability and evidence
    model_prior_probabilities = np.ones((len(variables_list),)) / len(variables_list)
    model_evidences = np.ones_like(model_prior_probabilities)

    logfile.write('\n\n==========================')
    logfile.write('\nLooping over each model')
    # For each model:
    for model_number, parameters_of_model in enumerate(variables_list):
        logfile.write('\n\n\t==========================')
        logfile.write(f'\n\tStarting analysis for model {model_number + 1}')
        logfile.write('\n\t==========================')

        # Assign probability distributions to the parameters of the model
        logfile.write('\n\t\tAssigning probability distributions to the parameters')
        all_distributions_list = make_distributions(variables=parameters_of_model)

        # Run the Algorithm
        logfile.write('\n\n\t==========================')
        logfile.write('\n\tRunning the TMCMC algorithm')
        logfile.write('\n\t==========================')

        # set the seed
        np.random.seed(tmcmc_data_instance.seedVal)
        logfile.write(f'\n\tSeed: {tmcmc_data_instance.seedVal}')

        syncLogFile(logfile)

        mytrace, log_evidence = run_TMCMC(  # noqa: F841
            number_of_samples,
            number_of_samples,
            all_distributions_list,
            number_of_MCMC_steps,
            max_number_of_MCMC_steps,
            # loglikelihood_module.log_likelihood,
            log_likelihood_function,
            parameters_of_model,
            working_directory,
            tmcmc_data_instance.seedVal,
            transformed_calibration_data,
            number_of_experiments,
            covariance_matrix_list,
            edp_names_list,
            edp_lengths_list,
            scale_factors,
            shift_factors,
            run_type,
            logfile,
            tmcmc_data_instance.MPI_size,
            driver_file,
            tmcmc_data_instance.parallelizeMCMC,
            model_number,
            total_number_of_models_in_ensemble,
        )
        logfile.write('\n\n\t==========================')
        logfile.write('\n\tTMCMC algorithm finished running')
        logfile.write('\n\t==========================')

        syncLogFile(logfile)

        logfile.write('\n\n\t==========================')
        logfile.write('\n\tStarting post-processing')

        # Compute model evidence
        logfile.write('\n\n\t\tComputing the model evidence')
        # evidence = 1
        # for i in range(len(mytrace)):
        #     Wm = mytrace[i][2]
        #     evidence *= np.mean(Wm)
        # logfile.write("\n\t\t\tModel evidence: {:g}".format(evidence))
        evidence = np.exp(log_evidence)
        logfile.write(f'\n\t\t\tModel evidence: {evidence:g}')
        logfile.write(f'\n\t\t\tModel log_evidence: {log_evidence:g}')

        syncLogFile(logfile)

        logfile.write('\n\n\t==========================')
        logfile.write('\n\tPost processing finished')
        logfile.write('\n\t==========================')

        syncLogFile(logfile)

        # Delete Analysis Folders

        # for analysisNumber in range(0, Np):
        #     stringToAppend = ("workdir." + str(analysisNumber + 1))
        #     analysisLocation = os.path.join(workdirMain, stringToAppend)
        #     # analysisPath = Path(analysisLocation)
        #     analysisPath = os.path.abspath(analysisLocation)
        #     shutil.rmtree(analysisPath)

        model_evidences[model_number] = evidence

        logfile.write('\n\n\t==========================')
        logfile.write(f'\n\tCompleted analysis for model {model_number + 1}')
        logfile.write('\n\t==========================')

        syncLogFile(logfile)

    modelPosteriorProbabilities = computeModelPosteriorProbabilities(  # noqa: N806
        model_prior_probabilities, model_evidences
    )

    logfile.write('\n\n==========================')
    logfile.write('\nFinished looping over each model')
    logfile.write('\n==========================\n')

    logfile.write('\nThe posterior model probabilities are:')
    for model_number in range(len(variables_list)):
        logfile.write(
            f'\nModel number {model_number + 1}: {modelPosteriorProbabilities[model_number] * 100:15g}%'
        )

    # ======================================================================================================================
    logfile.write('\nUCSD_UQ engine workflow complete!\n')
    logfile.write(f'\nTime taken: {(time.time() - t1) / 60:0.2f} minutes\n\n')

    syncLogFile(logfile)

    logfile.close()

    if run_type == 'runningRemote':
        tmcmc_data_instance.comm.Abort(0)

    # ======================================================================================================================


# ======================================================================================================================

if __name__ == '__main__':
    inputArgs = sys.argv  # noqa: N816
    main(inputArgs)

# ======================================================================================================================
