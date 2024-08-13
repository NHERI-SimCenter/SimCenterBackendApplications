# from scipy.stats import multivariate_normal  # noqa: CPY001, D100, INP001

# def log_likelihood(residuals, mean, cov):
#     return multivariate_normal.logpdf(residuals, mean=mean, cov=cov)

import numpy as np


def log_likelihood(residuals, mean, cov):  # noqa: ARG001, D103
    length = len(residuals)
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

    return ll
