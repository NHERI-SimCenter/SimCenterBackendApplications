"""
authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, Prof. J.P. Conte, Aakash Bangalore Satish*
affiliation: University of California, San Diego, *SimCenter, University of California, Berkeley

"""

# ======================================================================================================================
import os
import sys
import time
from shutil import copyfile
from typing import List, TextIO

# from importlib import import_module
import numpy as np

import pdfs
from parseData import parseDataFunction
from runTMCMC import RunTMCMC


# ======================================================================================================================

def createLogFile(workdirMain: str, logFileName: str):
    logFile = open(os.path.join(workdirMain, logFileName), "w")
    logFile.write(
        "Starting analysis at: {}".format(
            time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        )
    )
    logFile.write("\nRunning quoFEM's UCSD_UQ engine workflow")
    logFile.write("\nCWD: {}".format(os.path.abspath(".")))
    return logFile   


def syncLogFile(logFile: TextIO):
    logFile.flush()
    os.fsync(logFile.fileno())


def transformData(calibrationData: np.ndarray, edpLengthsList: List[int], scaleFactors: List[float], shiftFactors: List[float]):
    currentPosition = 0
    for j in range(len(edpLengthsList)):
        calibrationDataSlice = calibrationData[:, currentPosition : currentPosition + edpLengthsList[j]]
        calibrationDataSlice = calibrationDataSlice + shiftFactors[j]
        calibrationData[:, currentPosition : currentPosition + edpLengthsList[j]] = (calibrationDataSlice / scaleFactors[j])
        currentPosition += edpLengthsList[j]
    return calibrationData

# ======================================================================================================================

class DataProcessingError(Exception):
    """Raised when errors found when processing user-supplied calibration and covariance data.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


# ======================================================================================================================
class TransformData:
    def __init__(self, transformStrategy: str, logFile: TextIO) -> None:
        self.logFile = logFile
        self.transformStrategyList = ["absMaxScaling", "standardize"]
        if transformStrategy not in self.transformStrategyList:
            string = " or ".join(self.transformStrategyList)
            raise ValueError(f"transform strategy must be one of {string}")
        else:
            self.transformStrategy = transformStrategy
        
        logFile.write(
            "\n\nFor numerical convenience, a transformation is applied to the calibration data \nand model "
            "prediction corresponding to each response quantity. \nThe calibration data and model prediction for "
            "each response variable will \nfirst be shifted (a scalar value will be added to the data and "
            "prediction) and \nthen scaled (the data and prediction will be divided by a positive scalar value)."
        )

    def computeScaleAndShiftFactors(self, calibrationData: np.ndarray, edpLengthsList: List[int]):
        self.calibrationData = calibrationData
        self.edpLengthsList = edpLengthsList

        shiftFactors = []
        scaleFactors = []
        currentPosition = 0
        locShift = 0.0
        if self.transformStrategy in ["absMaxScaling"]:
            # Compute the scale factors - absolute maximum of the data for each response variable
            self.logFile.write(
                "\n\nComputing scale and shift factors. "
                "\n\tThe shift factors are set to 0.0 by default."
                "\n\tThe scale factors used are the absolute maximum of the data for each response variable."
                "\n\tIf the absolute maximum of the data for any response variable is 0.0, "
                "\n\tthen the scale factor is set to 1.0, and the shift factor is set to 1.0."
            )
            for j in range(len(self.edpLengthsList)):
                calibrationDataSlice = calibrationData[:, currentPosition : currentPosition + self.edpLengthsList[j]]
                absMax = np.absolute(np.max(calibrationDataSlice))
                if absMax == 0:  # This is to handle the case if abs max of data = 0.
                    locShift = 1.0
                    absMax = 1.0
                shiftFactors.append(locShift)
                scaleFactors.append(absMax)
                currentPosition += self.edpLengthsList[j]
        else:
            logFile.write(
            "\n\nComputing scale and shift factors. "
            "\n\tThe shift factors are set to the negative of the mean value for each response variable."
            "\n\tThe scale factors used are the standard deviation of the data for each response variable."
            "\n\tIf the standard deviation of the data for any response variable is 0.0, "
            "\n\tthen the scale factor is set to 1.0."
            )
            for j in range(len(self.edpLengthsList)):
                calibrationDataSlice = calibrationData[:, currentPosition : currentPosition + self.edpLengthsList[j]]
                meanValue = np.nanmean(calibrationDataSlice)
                stdValue = np.nanstd(calibrationDataSlice)
                if stdValue == 0:  # This is to handle the case if stdev of data = 0.
                    stdValue = 1.0
                shiftFactors.append(stdValue)
                scaleFactors.append(-meanValue)
                currentPosition += self.edpLengthsList[j]

        self.scaleFactors = scaleFactors
        self.shiftFactors = shiftFactors

    def transformDataMethod(self):
        return transformData(self.calibrationData, self.edpLengthsList, self.scaleFactors, self.shiftFactors)


# ======================================================================================================================
class CalDataPreparer:
    def __init__(self, workdirMain: str, workdirTemplate: str, calDataFileName: str, edpNamesList: List[str], edpLengthsList: List[int], logFile: TextIO) -> None:
        self.workdirMain = workdirMain
        self.workdirTemplate = workdirTemplate
        self.calDataFileName = calDataFileName
        self.edpNamesList = edpNamesList
        self.edpLengthsList = edpLengthsList
        self.logFile = logFile
        self.lineLength = sum(edpLengthsList)
        self.moveCalDataFile(self.calDataFileName)
    
    def moveCalDataFile(self, calDataFileName):
        os.rename(
            os.path.join(self.workdirTemplate, calDataFileName),
            os.path.join(self.workdirMain, calDataFileName),
        )

    def createHeadings(self):
        self.logFile.write("\n\tCreating headings")
        headings = "Exp_num interface "
        for i, edpName in enumerate(self.edpNamesList):
            if self.edpLengthsList[i] == 1:
                headings += "{} ".format(edpName)
            else:
                for comp in range(self.edpLengthsList[i]):
                    headings += "{}_{} ".format(edpName, comp + 1)
        self.logFile.write("\n\t\tThe headings are: \n\t\t{}".format(headings))
        return headings

    def createTempCalDataFile(self, calDataFile):
        self.tempCalDataFile = os.path.join(self.workdirMain, "quoFEMTempCalibrationDataFile.cal")
        f1 = open(self.tempCalDataFile, "w")
        headings = self.createHeadings()
        f1.write(headings)
        interface = 1
        self.numExperiments = 0
        linenum = 0
        with open(calDataFile, "r") as f:
            for line in f:
                linenum += 1
                if len(line.strip()) == 0:
                    continue
                else:
                    line = line.replace(",", " ")
                    # Check length of each line
                    words = line.split()
                    if len(words) == self.lineLength:
                        self.numExperiments += 1
                        tempLine = "{} {} ".format(self.numExperiments, interface)
                        for w in words:
                            tempLine += "{} ".format(w)
                        self.logFile.write(
                            "\n\tLine {}, length {}: \n\t\t{}".format(
                                linenum, len(words), tempLine
                            )
                        )
                        f1.write("\n{}".format(tempLine))
                    else:
                        self.logFile.write(
                            "\nERROR: The number of entries ({}) in line num {} of the file '{}' "
                            "does not match the expected length {}".format(
                                len(words), linenum, calDataFile, self.lineLength
                            )
                        )
                        raise DataProcessingError(
                            "ERROR: The number of entries ({}) in line num {} of the file '{}' "
                            "does not match the expected length {}".format(
                                len(words), linenum, calDataFile, self.lineLength
                            )
                        )
        f1.close()

    def readCleanedCalData(self):
        self.calibrationData = np.atleast_2d(
            np.genfromtxt(
                self.tempCalDataFile, skip_header=1, usecols=np.arange(2, 2 + self.lineLength)
            )
        )
    
    def getCalibrationData(self):
        calDataFile = os.path.join(self.workdirMain, self.calDataFileName)
        self.logFile.write(
            "\nCalibration data file being processed: \n\t{}\n".format(calDataFile)
        )
        self.createTempCalDataFile(calDataFile)
        self.readCleanedCalData()
        return self.calibrationData, self.numExperiments


# ======================================================================================================================

class TMCMCquoFEM:
    def __init__(self, mainscriptPath: str, workdirMain: str, runType: str, workflowDriver: str, logFile: TextIO) -> None:
        self.mainscriptPath = mainscriptPath
        self.workdirMain = workdirMain
        self.runType = runType
        self.workflowDriver = workflowDriver
        self.logFile = logFile

        self.MPI_size = self.getMPI_size()
        self.parallelizeMCMC = True


    def getMPI_size(self):
        if self.runType == "runningRemote":
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI_size = self.comm.Get_size()


    def syncLogFile(self):
        self.logFile.flush()
        os.fsync(self.logFile.fileno())


    def updateUQInfo(self, numberOfSamples, seedVal):
        self.numberOfSamples = numberOfSamples
        self.seedVal = seedVal

# ======================================================================================================================

class CovarianceMatrixPreparer:
    def __init__(self, calibrationData: np.ndarray, edpLengthsList: List[int], edpNamesList: List[str], workdirMain: str, numExperiments: int, logFile: TextIO) -> None:
        self.calibrationData = calibrationData
        self.edpLengthsList = edpLengthsList
        self.edpNamesList = edpNamesList
        self.workdirMain = workdirMain
        self.numExperiments = numExperiments
        self.logFile = logFile

        self.logFile.write("\n\n==========================")
        self.logFile.write("\nProcessing options for variance/covariance:")
        self.logFile.write(
            "\n\tOne variance value or covariance matrix will be used per response quantity per experiment."
        )
        self.logFile.write(
            "\n\tIf the user does not supply variance or covariance data, a default variance value will be\n\t"
            "used per response quantity, which is constant across experiments. The default variance is\n\t"
            "computed as the variance of the transformed data, if there is data from more than one "
            "experiment.\n\t"
            "If there is data from only one experiment, then a default variance value is computed by \n\t"
            "assuming that the standard deviation of the error is 5% of the absolute maximum value of \n\t"
            "the corresponding transformed response data."
        )

    def getDefaultErrorVariances(self):
        # For each response variable, compute the variance of the data. These will be the default error variance
        # values used in the calibration process. Values of the multiplier on these default error variance values will be
        # calibrated. There will be one such error variance value per response quantity. If there is only data from one
        # experiment,then the default error std.dev. value is assumed to be 5% of the absolute maximum value of the data
        # corresponding to that response quantity.
        defaultErrorVariances = np.zeros_like(self.edpLengthsList, dtype=float)
        if (np.shape(self.calibrationData)[0] > 1):  # if there are more than 1 rows of data, i.e. data from multiple experiments
            currentIndex = 0
            for i in range(len(self.edpLengthsList)):
                dataSlice = self.calibrationData[:, currentIndex : currentIndex + self.edpLengthsList[i]]
                defaultErrorVariances[i] = np.nanvar(dataSlice)
                currentIndex += self.edpLengthsList[i]
        else:
            currentIndex = 0
            for i in range(len(self.edpLengthsList)):
                dataSlice = self.calibrationData[:, currentIndex : currentIndex + self.edpLengthsList[i]]
                defaultErrorVariances[i] = (0.05 * np.max(np.absolute(dataSlice))) ** 2
                currentIndex += self.edpLengthsList[i]
        self.defaultErrorVariances = defaultErrorVariances
    
    def createCovarianceMatrix(self):
        covarianceMatrixList = []
        covarianceTypeList = []

        logFile = self.logFile
        edpNamesList = self.edpNamesList
        workdirMain = self.workdirMain
        numExperiments = self.numExperiments
        
        logFile.write("\n\nLooping over the experiments and EDPs")
        # First, check if the user has passed in any covariance matrix data
        for expNum in range(1, numExperiments + 1):
            logFile.write("\n\nExperiment number: {}".format(expNum))
            for i, edpName in enumerate(edpNamesList):
                logFile.write("\n\tEDP: {}".format(edpName))
                covarianceFileName = "{}.{}.sigma".format(edpName, expNum)
                covarianceFile = os.path.join(workdirMain, covarianceFileName)
                logFile.write(
                    "\n\t\tChecking to see if user-supplied file '{}' exists in '{}'".format(
                        covarianceFileName, workdirMain
                    )
                )
                if os.path.isfile(covarianceFile):
                    logFile.write("\n\t\tFound a user supplied file.")
                    if runType == "runningLocal":
                        src = covarianceFile
                        dst = os.path.join(workdirMain, covarianceFileName)
                        logFile.write(
                            "\n\t\tCopying user-supplied covariance file from {} to {}".format(
                                src, dst
                            )
                        )
                        copyfile(src, dst)
                        covarianceFile = dst
                    logFile.write(
                        "\n\t\tReading in user supplied covariance matrix from file: '{}'".format(
                            covarianceFile
                        )
                    )
                    # Check the data in the covariance matrix file
                    tmpCovFile = os.path.join(workdirMain, "quoFEMTempCovMatrixFile.sigma")
                    numRows = 0
                    numCols = 0
                    linenum = 0
                    with open(tmpCovFile, "w") as f1:
                        with open(covarianceFile, "r") as f:
                            for line in f:
                                linenum += 1
                                if len(line.strip()) == 0:
                                    continue
                                else:
                                    line = line.replace(",", " ")
                                    # Check the length of the line
                                    words = line.split()
                                    if numRows == 0:
                                        numCols = len(words)
                                    else:
                                        if numCols != len(words):
                                            logFile.write(
                                                "\nERROR: The number of columns in line {} do not match the "
                                                "number of columns in line {} of file {}.".format(
                                                    numRows, numRows - 1, covarianceFile
                                                )
                                            )
                                            raise DataProcessingError(
                                                "ERROR: The number of columns in line {} do not match the "
                                                "number of columns in line {} of file {}.".format(
                                                    numRows, numRows - 1, covarianceFile
                                                )
                                            )
                                    tempLine = ""
                                    for w in words:
                                        tempLine += "{} ".format(w)
                                    # logFile.write("\ncovMatrixLine {}: ".format(linenum), tempLine)
                                    if numRows == 0:
                                        f1.write(tempLine)
                                    else:
                                        f1.write("\n")
                                        f1.write(tempLine)
                                    numRows += 1
                    covMatrix = np.genfromtxt(tmpCovFile)
                    covarianceMatrixList.append(covMatrix)
                    # os.remove(tmpCovFile)
                    logFile.write(
                        "\n\t\tFinished reading the file. Checking the dimensions of the covariance data."
                    )
                    if numRows == 1:
                        if numCols == 1:
                            covarianceTypeList.append("scalar")
                            logFile.write(
                                "\n\t\tScalar variance value provided. The covariance matrix is an identity matrix "
                                "multiplied by this value."
                            )
                        elif numCols == edpLengthsList[i]:
                            covarianceTypeList.append("diagonal")
                            logFile.write(
                                "\n\t\tA row vector provided. This will be treated as the diagonal entries of the "
                                "covariance matrix."
                            )
                        else:
                            logFile.write(
                                "\nERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, edpLengthsList[i], numCols
                                )
                            )
                            raise DataProcessingError(
                                "ERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, edpLengthsList[i], numCols
                                )
                            )
                    elif numRows == edpLengthsList[i]:
                        if numCols == 1:
                            covarianceTypeList.append("diagonal")
                            logFile.write(
                                "\t\tA column vector provided. This will be treated as the diagonal entries of the "
                                "covariance matrix."
                            )
                        elif numCols == edpLengthsList[i]:
                            covarianceTypeList.append("matrix")
                            logFile.write("\n\t\tA full covariance matrix provided.")
                        else:
                            logFile.write(
                                "\nERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, edpLengthsList[i], numCols
                                )
                            )
                            raise DataProcessingError(
                                "ERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, edpLengthsList[i], numCols
                                )
                            )
                    else:
                        logFile.write(
                            "\nERROR: The number of rows of data in the covariance matrix file {}"
                            " must be either 1 or {}. Found {} rows".format(
                                covarianceFile, edpLengthsList[i], numCols
                            )
                        )
                        raise DataProcessingError(
                            "ERROR: The number of rows of data in the covariance matrix file {}"
                            " must be either 1 or {}. Found {} rows".format(
                                covarianceFile, edpLengthsList[i], numCols
                            )
                        )
                    logFile.write("\n\t\tCovariance matrix: {}".format(covMatrix))
                else:
                    logFile.write(
                        "\n\t\tDid not find a user supplied file. Using the default variance value."
                    )
                    logFile.write(
                        "\n\t\tThe covariance matrix is an identity matrix multiplied by this value."
                    )
                    scalarVariance = np.array(self.defaultErrorVariances[i])
                    covarianceMatrixList.append(scalarVariance)
                    covarianceTypeList.append("scalar")
                    logFile.write("\n\t\tCovariance matrix: {}".format(scalarVariance))
        self.covarianceMatrixList = covarianceMatrixList
        self.covarianceTypeList = covarianceTypeList
        return self.covarianceMatrixList
# ======================================================================================================================

if __name__ == "__main__":

    inputArgs = sys.argv

    mainscriptPath = os.path.abspath(inputArgs[0])
    workdirMain = os.path.abspath(inputArgs[1])
    workdirTemplate = os.path.abspath(inputArgs[2])
    runType = inputArgs[3]  # either "runningLocal" or "runningRemote"
    workflowDriver = inputArgs[4]
    inputFile = inputArgs[5]
    logFileName = "logFileTMCMC.txt"

    # # ================================================================================================================

    t1 = time.time()

    logFile = createLogFile(workdirMain, logFileName)

    # # ================================================================================================================

    inputJsonFilePath = os.path.join(os.path.abspath(workdirTemplate), inputFile)
    logFile.write("\n\n==========================")
    logFile.write("\nParsing the json input file {}".format(inputJsonFilePath))
    (numberOfSamples, seedVal, calDataFileName, logLikeModule, writeOutputs, variablesList, edpNamesList, 
    edpLengthsList) = parseDataFunction(inputJsonFilePath, logFile, workdirMain, os.path.dirname(mainscriptPath))
    syncLogFile(logFile)

    # # ================================================================================================================

    TMCMC = TMCMCquoFEM(mainscriptPath, workdirMain, runType, workflowDriver, logFile) 
    TMCMC.updateUQInfo(numberOfSamples, seedVal)   

    # # ================================================================================================================

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
    # Processing covariance matrix options
    CovMatrixOptions = CovarianceMatrixPreparer(transformedCalibrationData, edpLengthsList, edpNamesList, workdirMain, numExperiments, logFile)
    defaultErrorVariances = CovMatrixOptions.getDefaultErrorVariances()
    covarianceMatrixList = CovMatrixOptions.createCovarianceMatrix()

    # ======================================================================================================================
    # Starting TMCMC workflow
    logFile.write("\n\n==========================")
    logFile.write("\nSetting up the TMCMC algorithm")

    # sys.path.append(workdirMain)
    logFile.write("\n\tResults path: {}".format(workdirMain))

    # set the seed
    np.random.seed(TMCMC.seedVal)
    logFile.write("\n\tSeed: {}".format(TMCMC.seedVal))

    # number of particles: Np
    Np = TMCMC.numberOfSamples
    logFile.write("\n\tNumber of particles: {}".format(Np))

    # number of max MCMC steps
    Nm_steps_max = 10
    Nm_steps_maxmax = 10
    logFile.write("\n\tNumber of MCMC steps in first stage: {}".format(Nm_steps_max))
    logFile.write(
        "\n\tMax. number of MCMC steps in any stage: {}".format(Nm_steps_maxmax)
    )

    syncLogFile(logFile)

    # ======================================================================================================================
    logFile.write("\n\n==========================")
    logFile.write("\nLooping over each model")
    # For each model:
    for modelNum, variables in enumerate(variablesList):
        logFile.write("\n\n\t==========================")
        logFile.write("\n\tStarting analysis for model {}".format(modelNum))
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
                VariableValues = float(variables["Par1"][i])
                VariableWeights = float(variables["Par2"][i])

                AllPars.append(
                    pdfs.DiscreteDist(values=VariableValues, weights=VariableWeights)
                )

        # Run the Algorithm
        logFile.write("\n\n\t==========================")
        logFile.write("\n\tRunning the TMCMC algorithm")
        logFile.write("\n\t==========================")

        syncLogFile(logFile)

        mytrace, log_evidence = RunTMCMC(
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
            evidence = evidence * (sum(Wm) / len(Wm))
        logFile.write("\n\t\t\tModel evidence: {:e}".format(evidence))
        # evidence = np.exp(log_evidence)
        # logFile.write("\n\t\t\tModel evidence from log_evidence: {:e}".format(evidence))

        syncLogFile(logFile)

        # Write the results of the last stage to a file named dakotaTab.out for quoFEM to be able to read the results
        logFile.write(
            "\n\n\t\tWriting posterior samples to 'dakotaTab.out' for quoFEM to read the results"
        )
        tabFilePath = os.path.join(workdirMain, "dakotaTab.out")

        # Create the headings, which will be the first line of the file
        logFile.write("\n\t\t\tCreating headings")
        headings = "eval_id\tinterface\t"
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
        with open(tabFilePath, "w") as f:
            f.write(headings)
            for i in range(Np):
                string = "{}\t{}\t".format(i + 1, 1)
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

        logFile.write("\n\n\t==========================")
        logFile.write("\n\tCompleted analysis for model {}".format(modelNum))
        logFile.write("\n\t==========================")

        syncLogFile(logFile)

    logFile.write("\n\n==========================")
    logFile.write("\nFinished looping over each model")
    logFile.write("\n==========================\n")

    # ======================================================================================================================
    logFile.write("\nUCSD_UQ engine workflow complete!\n")
    logFile.write("\nTime taken: {:0.2f} minutes\n\n".format((time.time() - t1) / 60))

    syncLogFile(logFile)

    logFile.close()

    if runType == "runningRemote":
        TMCMC.comm.Abort(0)

    # ======================================================================================================================
