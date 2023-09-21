"""
authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, Prof. J.P. Conte, Aakash Bangalore Satish*
affiliation: University of California, San Diego, *SimCenter, University of California, Berkeley

"""

# ======================================================================================================================
import os
import sys
import time
from typing import TextIO
import numpy as np

import pdfs
from parseData import parseDataFunction
from runTMCMC import RunTMCMC
from utils import CovarianceMatrixPreparer, CalDataPreparer, TransformData, createLogFile, syncLogFile

# ======================================================================================================================

def computeModelPosteriorProbabilities(modelPriorProbabilities, modelEvidences):
    denominator = np.dot(modelPriorProbabilities, modelEvidences)
    return modelPriorProbabilities*modelEvidences/denominator

def computeModelPosteriorProbabilitiesUsingLogEvidences(modelPriorProbabilities, modelLogEvidences):
    deltas = modelLogEvidences - np.min(modelLogEvidences)
    denominator = np.dot(modelPriorProbabilities, np.exp(deltas))
    return modelPriorProbabilities*np.exp(deltas)/denominator

# ======================================================================================================================

class TMCMCInfo:
    def __init__(self, mainscriptPath: str, workdirMain: str, runType: str, workflowDriver: str, logFile: TextIO) -> None:
        self.mainscriptPath = mainscriptPath
        self.workdirMain = workdirMain
        self.runType = runType
        self.workflowDriver = workflowDriver
        self.logFile = logFile

        self.MPI_size = self.getMPI_size()
        self.parallelizeMCMC = True

        self.recommendedNumChains = 50
        self.numBurnInSteps = 10
        self.numSkipSteps = 1


    def getMPI_size(self):
        if self.runType == "runningRemote":
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI_size = self.comm.Get_size()

    # def syncLogFile(self):
    #     self.logFile.flush()
    #     os.fsync(self.logFile.fileno())


    def updateUQInfo(self, numberOfSamples, seedVal):
        self.numberOfSamples = numberOfSamples
        self.seedVal = seedVal
    
    def findNumProcessorsAvailable(self):
        if self.runType == "runningLocal":
            import multiprocessing as mp
            self.numProcessors = mp.cpu_count()
        elif self.runType == "runningRemote":
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.numProcessors = self.comm.Get_size()
        else:
            self.numProcessors = 1
    
    def getNumChains(self, numberOfSamples, runType, numProcessors):
        if runType == "runningLocal":
            self.numChains = int(min(numProcessors, self.recommendedNumChains))
        elif runType == "runningRemote":
            self.numChains = int(max(numProcessors, self.recommendedNumChains))
        else:
            self.numChains = self.recommendedNumChains
        
        if self.numChains < numberOfSamples:
            self.numChains = numberOfSamples
    
    def getNumStepsPerChainAfterBurnIn(self, numParticles, numChains):
        self.numStepsAfterBurnIn = int(np.ceil(numParticles/numChains)) * self.numSkipSteps
        # self.numStepsPerChain = numBurnInSteps + numStepsAfterBurnIn

    


# ======================================================================================================================

# ======================================================================================================================
def main(inputArgs):

    # Initialize analysis
    mainscriptPath = os.path.abspath(inputArgs[0])
    workdirMain = os.path.abspath(inputArgs[1])
    workdirTemplate = os.path.abspath(inputArgs[2])
    runType = inputArgs[3]  # either "runningLocal" or "runningRemote"
    workflowDriver = inputArgs[4]
    inputFile = inputArgs[5]
    logFileName = "logFileTMCMC.txt"
    try:
        os.remove('dakotaTab.out')
        os.remove('dakotTabPrior.out')
    except OSError:
        pass

    # # ================================================================================================================
    
    t1 = time.time()

    logFile = createLogFile(workdirMain, logFileName)

    # # ================================================================================================================

    # Process input json file
    inputJsonFilePath = os.path.join(os.path.abspath(workdirTemplate), inputFile)
    logFile.write("\n\n==========================")
    logFile.write("\nParsing the json input file {}".format(inputJsonFilePath))
    (numberOfSamples, seedVal, calDataFileName, logLikeModule, writeOutputs, variablesList, edpNamesList, 
    edpLengthsList, modelsDict, nModels) = parseDataFunction(inputJsonFilePath, logFile, workdirMain, 
    os.path.dirname(mainscriptPath))
    syncLogFile(logFile)

    # # ================================================================================================================

    # Initialize TMCMC object
    TMCMC = TMCMCInfo(mainscriptPath, workdirMain, runType, workflowDriver, logFile) 
    TMCMC.updateUQInfo(numberOfSamples, seedVal)  
    TMCMC.findNumProcessorsAvailable() 
    TMCMC.getNumChains(numberOfSamples, runType, TMCMC.numProcessors)
    TMCMC.getNumStepsPerChainAfterBurnIn(numberOfSamples, TMCMC.numChains)

    # # ================================================================================================================

    # Read calibration data
    DataPreparer = CalDataPreparer(workdirMain, workdirTemplate, calDataFileName, edpNamesList, edpLengthsList, logFile)
    calibrationData, numExperiments = DataPreparer.getCalibrationData()

    # # ================================================================================================================

    # Transform the data depending on the option chosen by the user
    transformation = "absMaxScaling"
    dataTransformer = TransformData(transformStrategy=transformation, logFile=logFile)
    dataTransformer.computeScaleAndShiftFactors(calibrationData, edpLengthsList)
    transformedCalibrationData = dataTransformer.transformDataMethod()
    scaleFactors = dataTransformer.scaleFactors
    shiftFactors = dataTransformer.shiftFactors

    logFile.write("\n\n\tThe scale and shift factors computed are: ")
    for j in range(len(edpNamesList)):
        logFile.write(
            "\n\t\tEDP: {}, scale factor: {}, shift factor: {}".format(
                edpNamesList[j], scaleFactors[j], shiftFactors[j]
            )
        )

    logFile.write("\n\nThe transformed calibration data: \n{}".format(transformedCalibrationData))

    # ======================================================================================================================
    # Process covariance matrix options
    CovMatrixOptions = CovarianceMatrixPreparer(transformedCalibrationData, edpLengthsList, edpNamesList, workdirMain, numExperiments, logFile, runType)
    defaultErrorVariances = CovMatrixOptions.getDefaultErrorVariances()
    covarianceMatrixList = CovMatrixOptions.createCovarianceMatrix()

    # ======================================================================================================================
    # Start TMCMC workflow
    logFile.write("\n\n==========================")
    logFile.write("\nSetting up the TMCMC algorithm")

    # sys.path.append(workdirMain)
    logFile.write("\n\tResults path: {}".format(workdirMain))

    # number of particles: Np
    Np = TMCMC.numberOfSamples
    logFile.write("\n\tNumber of particles: {}".format(Np))

    # number of max MCMC steps
    Nm_steps_max = TMCMC.numBurnInSteps + TMCMC.numStepsAfterBurnIn
    Nm_steps_maxmax = Nm_steps_max
    logFile.write("\n\tNumber of MCMC steps in first stage: {}".format(Nm_steps_max))
    logFile.write(
        "\n\tMax. number of MCMC steps in any stage: {}".format(Nm_steps_maxmax)
    )

    syncLogFile(logFile)

    # ======================================================================================================================
    # Initialize variables to store prior model probability and evidence
    modelPriorProbabilities = np.ones((len(variablesList),))/len(variablesList)
    modelEvidences = np.ones_like(modelPriorProbabilities)

    logFile.write("\n\n==========================")
    logFile.write("\nLooping over each model")
    # For each model:
    for modelNum, variables in enumerate(variablesList):
        logFile.write("\n\n\t==========================")
        logFile.write("\n\tStarting analysis for model {}".format(modelNum+1))
        logFile.write("\n\t==========================")
        # Assign probability distributions to the parameters
        logFile.write("\n\t\tAssigning probability distributions to the parameters")
        AllPars = []

        for i in range(len(variables["names"])):

            if variables["distributions"][i] == "Uniform":
                VariableLowerLimit = float(variables["Par1"][i])
                VariableUpperLimit = float(variables["Par2"][i])

                AllPars.append(
                    pdfs.Uniform(lower=VariableLowerLimit, upper=VariableUpperLimit)
                )

            if variables["distributions"][i] == "Normal":
                VariableMean = float(variables["Par1"][i])
                VariableSD = float(variables["Par2"][i])

                AllPars.append(pdfs.Normal(mu=VariableMean, sig=VariableSD))

            if variables["distributions"][i] == "Half-Normal":
                VariableSD = float(variables["Par1"][i])

                AllPars.append(pdfs.Halfnormal(sig=VariableSD))

            if variables["distributions"][i] == "Truncated-Normal":
                VariableMean = float(variables["Par1"][i])
                VariableSD = float(variables["Par2"][i])
                VariableLowerLimit = float(variables["Par3"][i])
                VariableUpperLimit = float(variables["Par4"][i])

                AllPars.append(
                    pdfs.TrunNormal(
                        mu=VariableMean,
                        sig=VariableSD,
                        a=VariableLowerLimit,
                        b=VariableUpperLimit,
                    )
                )

            if variables["distributions"][i] == "InvGamma":
                VariableA = float(variables["Par1"][i])
                VariableB = float(variables["Par2"][i])

                AllPars.append(pdfs.InvGamma(a=VariableA, b=VariableB))

            if variables["distributions"][i] == "Beta":
                VariableAlpha = float(variables["Par1"][i])
                VariableBeta = float(variables["Par2"][i])
                VariableLowerLimit = float(variables["Par3"][i])
                VariableUpperLimit = float(variables["Par4"][i])

                AllPars.append(
                    pdfs.BetaDist(
                        alpha=VariableAlpha,
                        beta=VariableBeta,
                        lowerbound=VariableLowerLimit,
                        upperbound=VariableUpperLimit,
                    )
                )

            if variables["distributions"][i] == "Lognormal":
                VariableMu = float(variables["Par1"][i])
                VariableSigma = float(variables["Par2"][i])

                AllPars.append(pdfs.LogNormDist(mu=VariableMu, sigma=VariableSigma))

            if variables["distributions"][i] == "Gumbel":
                VariableAlphaParam = float(variables["Par1"][i])
                VariableBetaParam = float(variables["Par2"][i])

                AllPars.append(
                    pdfs.GumbelDist(alpha=VariableAlphaParam, beta=VariableBetaParam)
                )

            if variables["distributions"][i] == "Weibull":
                VariableShapeParam = float(variables["Par1"][i])
                VariableScaleParam = float(variables["Par2"][i])

                AllPars.append(
                    pdfs.WeibullDist(shape=VariableShapeParam, scale=VariableScaleParam)
                )

            if variables["distributions"][i] == "Exponential":
                VariableLamda = float(variables["Par1"][i])

                AllPars.append(pdfs.ExponentialDist(lamda=VariableLamda))

            if variables["distributions"][i] == "Truncated exponential":
                VariableLamda = float(variables["Par1"][i])
                VariableLowerLimit = float(variables["Par2"][i])
                VariableUpperLimit = float(variables["Par3"][i])

                AllPars.append(
                    pdfs.TruncatedExponentialDist(
                        lamda=VariableLamda,
                        lower=VariableLowerLimit,
                        upper=VariableUpperLimit,
                    )
                )

            if variables["distributions"][i] == "Gamma":
                VariableK = float(variables["Par1"][i])
                VariableLamda = float(variables["Par2"][i])

                AllPars.append(pdfs.GammaDist(k=VariableK, lamda=VariableLamda))

            if variables["distributions"][i] == "Chisquare":
                VariableK = float(variables["Par1"][i])

                AllPars.append(pdfs.ChiSquareDist(k=VariableK))

            if variables["distributions"][i] == "Discrete":
                if variables["Par2"][i] is None:
                    VariableIndex = variables["Par1"][i]
                    AllPars.append(
                        pdfs.ConstantInteger(value=VariableIndex)
                    )
                else:
                    VariableValues = float(variables["Par1"][i])
                    VariableWeights = float(variables["Par2"][i])
                    AllPars.append(
                        pdfs.DiscreteDist(values=VariableValues, weights=VariableWeights)
                    )

        # Run the Algorithm
        logFile.write("\n\n\t==========================")
        logFile.write("\n\tRunning the TMCMC algorithm")
        logFile.write("\n\t==========================")

        # set the seed
        np.random.seed(TMCMC.seedVal)
        logFile.write("\n\tSeed: {}".format(TMCMC.seedVal))

        syncLogFile(logFile)

        mytrace, log_evidence = RunTMCMC(
            Np,
            Np,
            AllPars,
            Nm_steps_max,
            Nm_steps_maxmax,
            logLikeModule.log_likelihood,
            variables,
            workdirMain,
            TMCMC.seedVal,
            transformedCalibrationData,
            numExperiments,
            covarianceMatrixList,
            edpNamesList,
            edpLengthsList,
            scaleFactors,
            shiftFactors,
            runType,
            logFile,
            TMCMC.MPI_size,
            workflowDriver,
            TMCMC.parallelizeMCMC,
            modelNum,
            nModels
        )
        logFile.write("\n\n\t==========================")
        logFile.write("\n\tTMCMC algorithm finished running")
        logFile.write("\n\t==========================")

        syncLogFile(logFile)

        logFile.write("\n\n\t==========================")
        logFile.write("\n\tStarting post-processing")

        # Compute model evidence
        logFile.write("\n\n\t\tComputing the model evidence")
        evidence = 1
        for i in range(len(mytrace)):
            Wm = mytrace[i][2]
            evidence *= np.mean(Wm)
        logFile.write("\n\t\t\tModel evidence: {:g}".format(evidence))
        logFile.write("\n\t\t\tModel log_evidence: {:g}".format(log_evidence))

        syncLogFile(logFile)

        # Write the results of the last stage to a file named dakotaTab.out for quoFEM to be able to read the results
        logFile.write(
            "\n\n\t\tWriting posterior samples to 'dakotaTab.out' for quoFEM to read the results"
        )
        tabFilePath = os.path.join(workdirMain, "dakotaTab.out")

        # Create the headings, which will be the first line of the file
        headings = "eval_id\tinterface\t"
        if modelNum == 0:
            logFile.write("\n\t\t\tCreating headings")
            for v in variables["names"]:
                headings += "{}\t".format(v)
            if writeOutputs:  # create headings for outputs
                for i, edp in enumerate(edpNamesList):
                    if edpLengthsList[i] == 1:
                        headings += "{}\t".format(edp)
                    else:
                        for comp in range(edpLengthsList[i]):
                            headings += "{}_{}\t".format(edp, comp + 1)
            headings += "\n"

        # Get the data from the last stage
        logFile.write("\n\t\t\tGetting data from last stage")
        dataToWrite = mytrace[-1][0]

        logFile.write("\n\t\t\tWriting to file {}".format(tabFilePath))
        with open(tabFilePath, "a+") as f:
            if modelNum == 0:
                f.write(headings)
            for i in range(Np):
                string = "{}\t{}\t".format(i + 1 + Np*modelNum, modelNum+1)
                for j in range(len(variables["names"])):
                    string += "{}\t".format(dataToWrite[i, j])
                if writeOutputs:  # write the output data
                    analysisNumString = "workdir." + str(i + 1)
                    prediction = np.atleast_2d(
                        np.genfromtxt(
                            os.path.join(workdirMain, analysisNumString, "results.out")
                        )
                    ).reshape((1, -1))
                    for predNum in range(np.shape(prediction)[1]):
                        string += "{}\t".format(prediction[0, predNum])
                string += "\n"
                f.write(string)

        logFile.write("\n\n\t==========================")
        logFile.write("\n\tPost processing finished")
        logFile.write("\n\t==========================")

        syncLogFile(logFile)

        # Delete Analysis Folders

        # for analysisNumber in range(0, Np):
        #     stringToAppend = ("workdir." + str(analysisNumber + 1))
        #     analysisLocation = os.path.join(workdirMain, stringToAppend)
        #     # analysisPath = Path(analysisLocation)
        #     analysisPath = os.path.abspath(analysisLocation)
        #     shutil.rmtree(analysisPath)

        modelEvidences[modelNum] = evidence

        logFile.write("\n\n\t==========================")
        logFile.write("\n\tCompleted analysis for model {}".format(modelNum+1))
        logFile.write("\n\t==========================")

        syncLogFile(logFile)

    modelPosteriorProbabilities = computeModelPosteriorProbabilities(modelPriorProbabilities, modelEvidences)

    logFile.write("\n\n==========================")
    logFile.write("\nFinished looping over each model")
    logFile.write("\n==========================\n")

    logFile.write("\nThe posterior model probabilities are:")
    for modelNum in range(len(variablesList)):
        logFile.write(f"\nModel number {modelNum+1}: {modelPosteriorProbabilities[modelNum]*100:15g}%")

    # ======================================================================================================================
    logFile.write("\nUCSD_UQ engine workflow complete!\n")
    logFile.write("\nTime taken: {:0.2f} minutes\n\n".format((time.time() - t1) / 60))

    syncLogFile(logFile)

    logFile.close()

    if runType == "runningRemote":
        TMCMC.comm.Abort(0)

    # ======================================================================================================================


# ======================================================================================================================

if __name__ == "__main__":
    inputArgs = sys.argv
    main(inputArgs)

# ====================================================================================================================== 