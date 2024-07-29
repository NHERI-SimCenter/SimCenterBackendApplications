import numpy as np
from scipy.linalg import block_diag
import os
import shutil
import time
from numpy.typing import NDArray
from typing import Callable
from typing import TextIO
from importlib import import_module
import sys

import pdfs


class DataProcessingError(Exception):
    """Raised when errors found when processing user-supplied calibration and covariance data.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class CovarianceMatrixPreparer:
    def __init__(
        self,
        calibrationData: np.ndarray,
        edpLengthsList: list[int],
        edpNamesList: list[str],
        workdirMain: str,
        numExperiments: int,
        logFile: TextIO,
        runType: str,
    ) -> None:
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
        defaultErrorVariances = 1e-12 * np.ones_like(
            self.edpLengthsList, dtype=float
        )
        # defaultErrorVariances = np.zeros_like(self.edpLengthsList, dtype=float)
        if (
            np.shape(self.calibrationData)[0] > 1
        ):  # if there are more than 1 rows of data, i.e. data from multiple experiments
            currentIndex = 0
            for i in range(len(self.edpLengthsList)):
                dataSlice = self.calibrationData[
                    :, currentIndex : currentIndex + self.edpLengthsList[i]
                ]
                v = np.nanvar(dataSlice)
                if v != 0:
                    defaultErrorVariances[i] = v
                currentIndex += self.edpLengthsList[i]
        else:
            currentIndex = 0
            for i in range(len(self.edpLengthsList)):
                dataSlice = self.calibrationData[
                    :, currentIndex : currentIndex + self.edpLengthsList[i]
                ]
                v = np.max(np.absolute(dataSlice))
                if v != 0:
                    defaultErrorVariances[i] = (0.05 * v) ** 2
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
                    tmpCovFile = os.path.join(
                        workdirMain, "quoFEMTempCovMatrixFile.sigma"
                    )
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
                                                    numRows,
                                                    numRows - 1,
                                                    covarianceFile,
                                                )
                                            )
                                            raise DataProcessingError(
                                                "ERROR: The number of columns in line {} do not match the "
                                                "number of columns in line {} of file {}.".format(
                                                    numRows,
                                                    numRows - 1,
                                                    covarianceFile,
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
                                    covarianceFile,
                                    self.edpLengthsList[i],
                                    numCols,
                                )
                            )
                            raise DataProcessingError(
                                "ERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile,
                                    self.edpLengthsList[i],
                                    numCols,
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
                            logFile.write(
                                "\n\t\tA full covariance matrix provided."
                            )
                        else:
                            logFile.write(
                                "\nERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile,
                                    self.edpLengthsList[i],
                                    numCols,
                                )
                            )
                            raise DataProcessingError(
                                "ERROR: The number of columns of data in the covariance matrix file {}"
                                " must be either 1 or {}. Found {} columns".format(
                                    covarianceFile,
                                    self.edpLengthsList[i],
                                    numCols,
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
                    logFile.write(
                        "\n\t\tCovariance matrix: {}".format(covMatrix)
                    )
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
                    logFile.write(
                        "\n\t\tCovariance matrix: {}".format(scalarVariance)
                    )
        self.covarianceMatrixList = covarianceMatrixList
        self.covarianceTypeList = covarianceTypeList
        logFile.write(
            f"\n\nThe covariance matrix for prediction errors being used is:"
        )
        tmp = block_diag(*covarianceMatrixList)
        for row in tmp:
            rowString = " ".join([f"{col:14.8g}" for col in row])
            logFile.write("\n\t{}".format(rowString))
        return self.covarianceMatrixList


class CalDataPreparer:
    def __init__(
        self,
        workdirMain: str,
        workdirTemplate: str,
        calDataFileName: str,
        edpNamesList: list[str],
        edpLengthsList: list[int],
        logFile: TextIO,
    ) -> None:
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
        self.tempCalDataFile = os.path.join(
            self.workdirMain, "quoFEMTempCalibrationDataFile.cal"
        )
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
                        tempLine = "{} {} ".format(
                            self.numExperiments, interface
                        )
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
                                len(words),
                                linenum,
                                calDataFile,
                                self.lineLength,
                            )
                        )
                        raise DataProcessingError(
                            "ERROR: The number of entries ({}) in line num {} of the file '{}' "
                            "does not match the expected length {}".format(
                                len(words),
                                linenum,
                                calDataFile,
                                self.lineLength,
                            )
                        )
        f1.close()

    def readCleanedCalData(self):
        self.calibrationData = np.atleast_2d(
            np.genfromtxt(
                self.tempCalDataFile,
                skip_header=1,
                usecols=np.arange(2, 2 + self.lineLength).tolist(),
            )
        )

    def getCalibrationData(self):
        calDataFile = os.path.join(self.workdirMain, self.calDataFileName)
        self.logFile.write(
            "\nCalibration data file being processed: \n\t{}\n".format(
                calDataFile
            )
        )
        self.createTempCalDataFile(calDataFile)
        self.readCleanedCalData()
        return self.calibrationData, self.numExperiments


def transform_data_function(
    data_to_transform: np.ndarray,
    list_of_data_segment_lengths: list[int],
    list_of_scale_factors: list[float],
    list_of_shift_factors: list[float],
):
    currentPosition = 0
    for j in range(len(list_of_data_segment_lengths)):
        slice_of_data = data_to_transform[
            :,
            currentPosition : currentPosition
            + list_of_data_segment_lengths[j],
        ]
        slice_of_data = slice_of_data + list_of_shift_factors[j]
        data_to_transform[
            :,
            currentPosition : currentPosition
            + list_of_data_segment_lengths[j],
        ] = (
            slice_of_data / list_of_scale_factors[j]
        )
        currentPosition += list_of_data_segment_lengths[j]
    return data_to_transform


class DataTransformer:
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

    def computeScaleAndShiftFactors(
        self, calibrationData: np.ndarray, edpLengthsList: list[int]
    ):
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
                calibrationDataSlice = calibrationData[
                    :,
                    currentPosition : currentPosition + self.edpLengthsList[j],
                ]
                absMax = np.absolute(np.max(calibrationDataSlice))
                if (
                    absMax == 0
                ):  # This is to handle the case if abs max of data = 0.
                    locShift = 1.0
                    absMax = 1.0
                shiftFactors.append(locShift)
                scaleFactors.append(absMax)
                currentPosition += self.edpLengthsList[j]
        else:
            self.logFile.write(
                "\n\nComputing scale and shift factors. "
                "\n\tThe shift factors are set to the negative of the mean value for each response variable."
                "\n\tThe scale factors used are the standard deviation of the data for each response variable."
                "\n\tIf the standard deviation of the data for any response variable is 0.0, "
                "\n\tthen the scale factor is set to 1.0."
            )
            for j in range(len(self.edpLengthsList)):
                calibrationDataSlice = calibrationData[
                    :,
                    currentPosition : currentPosition + self.edpLengthsList[j],
                ]
                meanValue = np.nanmean(calibrationDataSlice)
                stdValue = np.nanstd(calibrationDataSlice)
                if (
                    stdValue == 0
                ):  # This is to handle the case if stdev of data = 0.
                    stdValue = 1.0
                scaleFactors.append(stdValue)
                shiftFactors.append(-meanValue)
                currentPosition += self.edpLengthsList[j]

        self.scaleFactors = scaleFactors
        self.shiftFactors = shiftFactors
        return scaleFactors, shiftFactors

    def transformData(self):
        return transform_data_function(
            self.calibrationData,
            self.edpLengthsList,
            self.scaleFactors,
            self.shiftFactors,
        )


def createLogFile(where: str, logfile_name: str):
    logfile = open(os.path.join(where, logfile_name), "w")
    logfile.write(
        "Starting analysis at: {}".format(
            time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        )
    )
    logfile.write("\nRunning quoFEM's UCSD_UQ engine workflow")
    logfile.write("\nCWD: {}".format(os.path.abspath(".")))
    return logfile


def syncLogFile(logFile: TextIO):
    logFile.flush()
    os.fsync(logFile.fileno())


def make_distributions(variables):

    all_distributions_list = []

    for i in range(len(variables["names"])):

        if variables["distributions"][i] == "Uniform":
            lower_limit = float(variables["Par1"][i])
            upper_limit = float(variables["Par2"][i])

            all_distributions_list.append(
                pdfs.Uniform(lower=lower_limit, upper=upper_limit)
            )

        if variables["distributions"][i] == "Normal":
            mean = float(variables["Par1"][i])
            standard_deviation = float(variables["Par2"][i])

            all_distributions_list.append(
                pdfs.Normal(mu=mean, sig=standard_deviation)
            )

        if variables["distributions"][i] == "Half-Normal":
            standard_deviation = float(variables["Par1"][i])

            all_distributions_list.append(
                pdfs.Halfnormal(sig=standard_deviation)
            )

        if variables["distributions"][i] == "Truncated-Normal":
            mean = float(variables["Par1"][i])
            standard_deviation = float(variables["Par2"][i])
            lower_limit = float(variables["Par3"][i])
            upper_limit = float(variables["Par4"][i])

            all_distributions_list.append(
                pdfs.TrunNormal(
                    mu=mean,
                    sig=standard_deviation,
                    a=lower_limit,
                    b=upper_limit,
                )
            )

        if variables["distributions"][i] == "InvGamma":
            a = float(variables["Par1"][i])
            b = float(variables["Par2"][i])

            all_distributions_list.append(pdfs.InvGamma(a=a, b=b))

        if variables["distributions"][i] == "Beta":
            alpha = float(variables["Par1"][i])
            beta = float(variables["Par2"][i])
            lower_limit = float(variables["Par3"][i])
            upper_limit = float(variables["Par4"][i])

            all_distributions_list.append(
                pdfs.BetaDist(
                    alpha=alpha,
                    beta=beta,
                    lowerbound=lower_limit,
                    upperbound=upper_limit,
                )
            )

        if variables["distributions"][i] == "Lognormal":
            mu = float(variables["Par1"][i])
            sigma = float(variables["Par2"][i])

            all_distributions_list.append(pdfs.LogNormDist(mu=mu, sigma=sigma))

        if variables["distributions"][i] == "Gumbel":
            alpha = float(variables["Par1"][i])
            beta = float(variables["Par2"][i])

            all_distributions_list.append(
                pdfs.GumbelDist(alpha=alpha, beta=beta)
            )

        if variables["distributions"][i] == "Weibull":
            shape = float(variables["Par1"][i])
            scale = float(variables["Par2"][i])

            all_distributions_list.append(
                pdfs.WeibullDist(shape=shape, scale=scale)
            )

        if variables["distributions"][i] == "Exponential":
            lamda = float(variables["Par1"][i])

            all_distributions_list.append(pdfs.ExponentialDist(lamda=lamda))

        if variables["distributions"][i] == "Truncated exponential":
            lamda = float(variables["Par1"][i])
            lower_limit = float(variables["Par2"][i])
            upper_limit = float(variables["Par3"][i])

            all_distributions_list.append(
                pdfs.TruncatedExponentialDist(
                    lamda=lamda,
                    lower=lower_limit,
                    upper=upper_limit,
                )
            )

        if variables["distributions"][i] == "Gamma":
            k = float(variables["Par1"][i])
            lamda = float(variables["Par2"][i])

            all_distributions_list.append(pdfs.GammaDist(k=k, lamda=lamda))

        if variables["distributions"][i] == "Chisquare":
            k = float(variables["Par1"][i])

            all_distributions_list.append(pdfs.ChiSquareDist(k=k))

        if variables["distributions"][i] == "Discrete":
            if variables["Par2"][i] is None:
                value = variables["Par1"][i]
                all_distributions_list.append(
                    pdfs.ConstantInteger(value=value)
                )
            else:
                values = float(variables["Par1"][i])
                weights = float(variables["Par2"][i])
                all_distributions_list.append(
                    pdfs.DiscreteDist(values=values, weights=weights)
                )

    return all_distributions_list


class LogLikelihoodHandler:
    def __init__(
        self,
        data: NDArray,
        covariance_matrix_blocks_list: list[NDArray],
        list_of_data_segment_lengths: list[int],
        list_of_scale_factors: list[float],
        list_of_shift_factors: list[float],
        workdir_main,
        full_path_to_tmcmc_code_directory: str,
        log_likelihood_file_name: str = "",
    ) -> None:
        self.data = data
        self.covariance_matrix_list = covariance_matrix_blocks_list
        self.list_of_data_segment_lengths = list_of_data_segment_lengths
        self.list_of_scale_factors = list_of_scale_factors
        self.list_of_shift_factors = list_of_shift_factors
        self.workdir_main = workdir_main
        self.full_path_to_tmcmc_code_directory = (
            full_path_to_tmcmc_code_directory
        )
        self.log_likelihood_file_name = log_likelihood_file_name
        sys.path.append(self.workdir_main)
        self._copy_log_likelihood_module()
        self.num_experiments = self._get_num_experiments()
        self.num_response_quantities = self._get_num_response_quantities()
        self.log_likelihood_function = self.get_log_likelihood_function()

    def _copy_log_likelihood_module(self):
        if (
            len(self.log_likelihood_file_name) == 0
        ):  # if the log-likelihood file is an empty string
            self.log_likelihood_file_name = "defaultLogLikeScript.py"
            src = os.path.join(
                self.full_path_to_tmcmc_code_directory,
                self.log_likelihood_file_name,
            )
            dst = os.path.join(
                self.workdir_main, self.log_likelihood_file_name
            )
            try:
                shutil.copyfile(src, dst)
            except Exception:
                msg = f"ERROR: The log-likelihood script '{src}' cannot be copied to '{dst}'."
                raise Exception(msg)

    def _get_num_experiments(self) -> int:
        return np.shape(self.data)[0]

    def _get_num_response_quantities(self) -> int:
        return len(self.list_of_data_segment_lengths)

    def _import_log_likelihood_module(
        self, log_likelihood_module_name: str
    ) -> Callable:
        try:
            module = import_module(log_likelihood_module_name)
        except:
            msg = "\n\t\t\t\tERROR: The log-likelihood script '{}' cannot be imported.".format(
                os.path.join(self.workdir_main, self.log_likelihood_file_name)
            )
            raise ImportError(msg)
        return module  # type: ignore

    def get_log_likelihood_function(self) -> Callable:
        log_likelihood_module_name = os.path.splitext(
            self.log_likelihood_file_name
        )[0]
        module = self._import_log_likelihood_module(log_likelihood_module_name)
        return module.log_likelihood

    def _transform_prediction(self, prediction: NDArray) -> NDArray:
        return transform_data_function(
            prediction,
            self.list_of_data_segment_lengths,
            self.list_of_scale_factors,
            self.list_of_shift_factors,
        )

    def _compute_residuals(self, transformed_prediction: NDArray) -> NDArray:
        return transformed_prediction - self.data

    def _make_mean(self, response_num: int) -> NDArray:
        return np.zeros((self.list_of_data_segment_lengths[response_num]))

    def _make_covariance(self, response_num, cov_multiplier) -> NDArray:
        return cov_multiplier * np.atleast_2d(
            self.covariance_matrix_list[response_num]
        )

    def _make_input_for_log_likelihood_function(self, prediction) -> list:
        return [
            self._transform_prediction(prediction),
        ]

    def _loop_for_log_likelihood(
        self,
        prediction,
        list_of_covariance_multipliers,
    ):
        transformed_prediction = self._transform_prediction(prediction)
        allResiduals = self._compute_residuals(transformed_prediction)
        loglike = 0
        for i in range(self.num_experiments):
            currentPosition = 0
            for j in range(self.num_response_quantities):
                length = self.list_of_data_segment_lengths[j]
                residuals = allResiduals[
                    i, currentPosition : currentPosition + length
                ]
                currentPosition = currentPosition + length
                cov = self._make_covariance(
                    j, list_of_covariance_multipliers[j]
                )
                mean = self._make_mean(j)
                ll = self.log_likelihood_function(residuals, mean, cov)
                if not np.isnan(ll):
                    loglike += ll
                else:
                    loglike += -np.inf
        return loglike

    def evaluate_log_likelihood(
        self, prediction: NDArray, list_of_covariance_multipliers: list[float]
    ) -> float:
        return self._loop_for_log_likelihood(
            prediction=prediction,
            list_of_covariance_multipliers=list_of_covariance_multipliers,
        )
