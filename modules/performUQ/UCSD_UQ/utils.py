import numpy as np
from scipy.linalg import block_diag
import os
import shutil

from typing import List, TextIO


class DataProcessingError(Exception):
    """Raised when errors found when processing user-supplied calibration and covariance data.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class CovarianceMatrixPreparer:
    def __init__(self, calibrationData: np.ndarray, edpLengthsList: List[int], edpNamesList: List[str], workdirMain: str, numExperiments: int, logFile: TextIO, runType: str) -> None:
        self.calibrationData = calibrationData
        self.edpLengthsList = edpLengthsList
        self.edpNamesList = edpNamesList
        self.workdirMain = workdirMain
        self.numExperiments = numExperiments
        self.logFile = logFile
        self.runType = runType

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
                    if self.runType == "runningLocal":
                        src = covarianceFile
                        dst = os.path.join(workdirMain, covarianceFileName)
                        logFile.write(
                            "\n\t\tCopying user-supplied covariance file from {} to {}".format(
                                src, dst
                            )
                        )
                        shutil.copyfile(src, dst)
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
                        elif numCols == self.edpLengthsList[i]:
                            covarianceTypeList.append("diagonal")
                            logFile.write(
                                "\n\t\tA row vector provided. This will be treated as the diagonal entries of the "
                                "covariance matrix."
                            )
                        else:
                            logFile.write(
                                "\nERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, self.edpLengthsList[i], numCols
                                )
                            )
                            raise DataProcessingError(
                                "ERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, self.edpLengthsList[i], numCols
                                )
                            )
                    elif numRows == self.edpLengthsList[i]:
                        if numCols == 1:
                            covarianceTypeList.append("diagonal")
                            logFile.write(
                                "\t\tA column vector provided. This will be treated as the diagonal entries of the "
                                "covariance matrix."
                            )
                        elif numCols == self.edpLengthsList[i]:
                            covarianceTypeList.append("matrix")
                            logFile.write("\n\t\tA full covariance matrix provided.")
                        else:
                            logFile.write(
                                "\nERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, self.edpLengthsList[i], numCols
                                )
                            )
                            raise DataProcessingError(
                                "ERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile, self.edpLengthsList[i], numCols
                                )
                            )
                    else:
                        logFile.write(
                            "\nERROR: The number of rows of data in the covariance matrix file {}"
                            " must be either 1 or {}. Found {} rows".format(
                                covarianceFile, self.edpLengthsList[i], numCols
                            )
                        )
                        raise DataProcessingError(
                            "ERROR: The number of rows of data in the covariance matrix file {}"
                            " must be either 1 or {}. Found {} rows".format(
                                covarianceFile, self.edpLengthsList[i], numCols
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
        logFile.write(f"\n\nThe covariance matrix for prediction errors being used is:")
        tmp = block_diag(*covarianceMatrixList)
        for row in tmp:
            rowString = " ".join([f"{col:14.8g}" for col in row])
            logFile.write("\n\t{}".format(rowString))
        return self.covarianceMatrixList



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
