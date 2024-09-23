import numpy as np  # noqa: INP001, D100


class CovError(Exception):
    """Raised when the number of covariance matrix terms are incorrect.

    Attributes
    ----------
        message -- explanation of the error

    """

    def __init__(self, message):
        self.message = message


def log_likelihood(
    calibrationData,  # noqa: N803
    prediction,
    numExperiments,  # noqa: N803
    covarianceMatrixList,  # noqa: N803
    edpNamesList,  # noqa: ARG001, N803
    edpLengthsList,  # noqa: N803
    covarianceMultiplierList,  # noqa: N803
    scaleFactors,  # noqa: N803
    shiftFactors,  # noqa: N803
):
    """Compute the log-likelihood

    :param calibrationData: Calibration data consisting of the measured values of response. Each row contains the data
    from one experiment. The length of each row equals the sum of the lengths of all response quantities.
    :type calibrationData: numpy ndarray (atleast_2d)

    :param prediction: Prediction of the response from the model, evaluated using the parameter values for
    which the log-likelihood function needs to be calculated.
    :type prediction: numpy ndarray (atleast_2d)

    :param numExperiments: Number of experiments from which data is available, this is equal to the number of rows
    (i.e., the first index) of the calibration data array
    :type numExperiments: int

    :param covarianceMatrixList: A list of length numExperiments * numResponses, where each item in the list contains
    the covariacne matrix or variance value corresponding to that experiment and response quantity
    :type covarianceMatrixList: list of numpy ndarrays

    :param edpNamesList: A list containing the names of the response quantities
    :type edpNamesList: list of strings

    :param edpLengthsList: A list containing the length of each response quantity
    :type edpLengthsList: list of ints

    :param covarianceMultiplierList: A list containing the covariance matrices or variance values. The length of this
    list is equal to the product of the number of experiments and the number of response quantities.
    :type covarianceMultiplierList: list of numpy ndarrays

    :param scaleFactors: A list containing the normalizing factors used to scale (i.e. divide) the model
    prediction values. The length of this list is equal to the number of response quantities.
    :type scaleFactors: list of ints

    :param shiftFactors: A list containing the values used to shift the prediction values. The locShift values are 0.0,
    unless the abs max of the data of that response quantity is 0. In this case, the locShift = 1.0. LocShift values
    must be added to the response quantities since they are added to the data. The length of this list is equal to the
    number of response quantities.
    :type shiftFactors: list of ints

    :return: loglikelihood. This is a scalar value, which is equal to the logpdf of a zero-mean multivariate normal
    distribution and a user-supplied covariance structure. Block-diagonal covariance structures are supported. The value
    of multipliers on the covariance block corresponding to each response quantity is also calibrated.
    :rtype: float
    """  # noqa: D400
    # Check if the correct number of covariance terms has been passed in
    numResponses = len(edpLengthsList)  # noqa: N806
    if len(covarianceMatrixList) != numExperiments * numResponses:
        print(  # noqa: T201
            f'ERROR: The expected number of covariance matrices is {numExperiments * numResponses}, but only {len(covarianceMatrixList)} were passed '
            'in.'
        )
        raise CovError(  # noqa: TRY003
            f'ERROR: The expected number of covariance matrices is {numExperiments * numResponses}, but only {len(covarianceMatrixList)} were passed '  # noqa: EM102
            'in.'
        )

    # Shift and normalize the prediction
    currentPosition = 0  # noqa: N806
    for j in range(len(edpLengthsList)):
        prediction[:, currentPosition : currentPosition + edpLengthsList[j]] = (
            prediction[:, currentPosition : currentPosition + edpLengthsList[j]]
            + shiftFactors[j]
        )
        prediction[:, currentPosition : currentPosition + edpLengthsList[j]] = (
            prediction[:, currentPosition : currentPosition + edpLengthsList[j]]
            / scaleFactors[j]
        )
        currentPosition = currentPosition + edpLengthsList[j]  # noqa: N806

    # Compute the normalized residuals
    allResiduals = prediction - calibrationData  # noqa: N806

    # Loop over the normalized residuals to compute the log-likelihood
    loglike = 0
    covListIndex = 0  # noqa: N806
    for i in range(numExperiments):
        currentPosition = 0  # noqa: N806
        for j in range(numResponses):
            # Get the residuals corresponding to this response variable
            length = edpLengthsList[j]
            residuals = allResiduals[i, currentPosition : currentPosition + length]
            currentPosition = currentPosition + length  # noqa: N806

            # Get the covariance matrix corresponding to this response variable
            cov = np.atleast_2d(covarianceMatrixList[covListIndex])
            covListIndex = covListIndex + 1  # noqa: N806

            # Multiply the covariance matrix by the value of the covariance multiplier
            cov = cov * covarianceMultiplierList[j]

            if np.shape(cov)[0] == np.shape(cov)[1] == 1:
                # If there is a single variance value that is constant for all residual terms, then this is the case of
                # having a sample of i.i.d. zero-mean normally distributed observations, and the log-likelihood can be
                # computed more efficiently
                var = cov[0][0]
                ll = (
                    -length / 2 * np.log(var)
                    - length / 2 * np.log(2 * np.pi)
                    - 1 / (2 * var) * np.sum(residuals**2)
                )
            else:
                if np.shape(cov)[0] != np.shape(cov)[1]:
                    cov = np.diag(cov.flatten())
                # The multivariate normal log-pdf is made up of three terms:
                # logpdf = -1/2*[(d*log(2*pi)) + (log(abs(det(cov)))) + (residual.T * inverse(cov) * residual) i.e.,
                # Mahalanobis distance]
                #                = -1/2*[t1 + t2 + t3]
                t1 = length * np.log(2 * np.pi)
                eigenValues, eigenVectors = np.linalg.eigh(cov)  # noqa: N806
                logdet = np.sum(np.log(eigenValues))
                eigenValuesReciprocal = 1.0 / eigenValues  # noqa: N806
                z = eigenVectors * np.sqrt(eigenValuesReciprocal)
                mahalanobisDistance = np.square(np.dot(residuals, z)).sum()  # noqa: N806
                ll = -0.5 * (t1 + logdet + mahalanobisDistance)
            if not np.isnan(ll):
                loglike += ll
            else:
                loglike += -np.inf
    return loglike  # noqa: DOC201, RUF100
