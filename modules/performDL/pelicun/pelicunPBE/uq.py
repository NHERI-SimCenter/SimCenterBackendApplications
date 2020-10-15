# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of pelicun.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay

"""
This module defines constants, classes and methods for uncertainty
quantification in pelicun.

.. rubric:: Contents

.. autosummary::

    RandomVariable
    RandomVariableSubset

    tmvn_rvs
    mvn_orthotope_density
    tmvn_MLE


"""

from .base import *

import warnings
from scipy.stats import norm, truncnorm, multivariate_normal, multinomial
from scipy.stats.mvn import mvndst
from scipy.optimize import minimize, differential_evolution
from copy import deepcopy

def tmvn_rvs(mu, COV, lower=None, upper=None, size=1):
    """
    Sample a truncated MVN distribution.

    Truncation of the multivariate normal distribution is currently considered
    through rejection sampling. The applicability of this method is limited by
    the amount of probability density enclosed by the hyperrectangle defined by
    the truncation limits. The lower that density is, the more samples will
    need to be rejected which makes the method inefficient when the tails of
    the MVN shall be sampled in high-dimensional space. Such cases can be
    handled by a Gibbs sampler, which is a planned future feature of this
    function.

    Parameters
    ----------
    mu: float scalar or ndarray
        Mean(s) of the non-truncated distribution.
    COV: float ndarray
        Covariance matrix of the non-truncated distribution.
    lower: float vector, optional, default: None
        Lower bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from below
        in a subset of the dimensions, assign an infinite value
        (i.e. -numpy.inf) to those dimensions.
    upper: float vector, optional, default: None
        Upper bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from above
        in a subset of the dimensions, assign an infinite value
        (i.e. numpy.inf) to those dimensions.
    size: int
        Number of samples requested.

    Returns
    -------
    samples: float ndarray
        Samples generated from the truncated distribution.

    """

    mu = np.asarray(mu)
    if mu.shape == ():
        mu = np.asarray([mu])
        COV = np.asarray([COV])

    # if there are no bounds, simply sample an MVN distribution
    if lower is None and upper is None:

        samples = multivariate_normal.rvs(mean=mu, cov=COV, size=size)

    else:
        # first, get the rejection rate
        alpha, eps_alpha = mvn_orthotope_density(mu, COV, lower, upper)

        # initialize the data for sample collection
        sample_count = 0
        samples = None
        ndim = len(mu)

        if lower is None:
            lower = np.ones(ndim) * -np.inf
        if upper is None:
            upper = np.ones(ndim) * np.inf

        # If the error in the alpha estimate is too large, then we are
        # beyond the applicability limits of the function used for
        # estimating alpha. Raise an error in such a case
        if alpha <= 100. * eps_alpha:  # i.e. max. error is limited at 1%
            #print(alpha, eps_alpha)
            raise ValueError(
                "The density of the joint probability distribution within the "
                "truncation limits is too small and cannot be estimated with "
                "sufficiently high accuracy. This is most probably due to "
                "incorrect limits set for the distribution."
            )

        # If the rejection rate is sufficiently low, perform rejection sampling
        # Note: the minimum rate is set to zero until a Gibbs sampler is
        # implemented, but a warning message is displayed for anything below
        # 1e-3
        if alpha < 1e-3:
            warnings.warn(UserWarning(
                "The rejection rate for sampling the prescribed truncated MVN "
                "distribution is higher than 0.999. This makes sampling with "
                "our current implementation very resource-intensive and "
                "inefficient. If you need to sample such parts of MVN "
                "distributions, please let us know and we will improve "
                "this function in the future."
            ))
        if alpha > 0.:
            while sample_count < size:

                # estimate the required number of samples
                req_samples = max(int(1.1*(size-sample_count)/alpha), 2)

                # generate the raw samples
                raw_samples = multivariate_normal.rvs(mu, COV,
                                                      size=req_samples)

                # remove the samples that are outside the truncation limits
                good_ones = np.all([raw_samples>lower, raw_samples<upper],
                                   axis=0)
                if ndim > 1:
                    good_ones = np.all(good_ones, axis=1)

                new_samples = raw_samples[good_ones]

                # add the new samples to the pool of samples
                if sample_count > 0:
                    samples = np.concatenate([samples, new_samples], axis=0)
                else:
                    samples = new_samples

                # check the number of available samples and generate more if
                # needed
                sample_count = samples.shape[0]

            samples = samples[:size]

        #else:
        # TODO: Gibbs sampler

    return samples

def mvn_orthotope_density(mu, COV, lower=None, upper=None):
    """
    Estimate the probability density within a hyperrectangle for an MVN distr.

    Use the method of Alan Genz (1992) to estimate the probability density
    of a multivariate normal distribution within an n-orthotope (i.e.,
    hyperrectangle) defined by its lower and upper bounds. Limits can be
    relaxed in any direction by assigning infinite bounds (i.e. numpy.inf).

    Parameters
    ----------
    mu: float scalar or ndarray
        Mean(s) of the non-truncated distribution.
    COV: float ndarray
        Covariance matrix of the non-truncated distribution
    lower: float vector, optional, default: None
        Lower bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from below
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. -numpy.inf) to those dimensions.
    upper: float vector, optional, default: None
        Upper bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from above
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. numpy.inf) to those dimensions.
    Returns
    -------
    alpha: float
        Estimate of the probability density within the hyperrectangle
    eps_alpha: float
        Estimate of the error in alpha.

    """

    # process the inputs and get the number of dimensions
    mu = np.asarray(mu)
    if mu.shape == ():
        mu = np.asarray([mu])
        COV = np.asarray([COV])
    else:
        COV = np.asarray(COV)
    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig,sig)

    ndim = len(mu)

    if lower is None:
        lower = -np.ones(ndim) * np.inf
    else:
        lower = np.asarray(lower)

    if upper is None:
        upper = np.ones(ndim) * np.inf
    else:
        upper = np.asarray(upper)

    # standardize the truncation limits
    lower = (lower-mu)/sig
    upper = (upper-mu)/sig

    # prepare the flags for infinite bounds (these are needed for the mvndst
    # function)
    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)
    infin = 2.0*np.ones(ndim)

    np.putmask(infin, lowinf, 0)
    np.putmask(infin, uppinf, 1)
    np.putmask(infin, lowinf*uppinf, -1)

    # prepare the correlation coefficients
    if ndim == 1:
        correl = 0
    else:
        correl = corr[np.tril_indices(ndim, -1)]

    # estimate the density
    eps_alpha, alpha, __ = mvndst(lower, upper, infin, correl)

    return alpha, eps_alpha

def tmvn_MLE(samples,
             tr_lower=None, tr_upper=None,
             censored_count=0, det_lower=None, det_upper=None,
             alpha_lim=None):
    """
    Fit a truncated multivariate normal distribution to samples using MLE.

    The number of dimensions of the distribution function are inferred from the
    shape of the sample data. Censoring is automatically considered if the
    number of censored samples and the corresponding detection limits are
    provided. Infinite or unspecified truncation limits lead to fitting a
    non-truncated normal distribution in that dimension.

    Parameters
    ----------
    samples: ndarray
        Raw data that serves as the basis of estimation. The number of samples
        equals the number of columns and each row introduces a new feature. In
        other words: a list of sample lists is expected where each sample list
        is a collection of samples of one variable.
    tr_lower: float vector, optional, default: None
        Lower bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from below
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. -numpy.inf) to those dimensions.
    tr_upper: float vector, optional, default: None
        Upper bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from above
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. numpy.inf) to those dimensions.
    censored_count: int, optional, default: None
        The number of censored samples that are beyond the detection limits.
        All samples outside the detection limits are aggregated into one set.
        This works the same way in one and in multiple dimensions. Prescription
        of specific censored sample counts for sub-regions of the input space
        outside the detection limits is not supported.
    det_lower: float ndarray, optional, default: None
        Lower detection limit(s) for censored data. In multivariate cases the
        limits need to be defined as a vector; a scalar value is sufficient in
        a univariate case. If the data is not censored from below in a
        particular dimension, assign None to that position of the ndarray.
    det_upper: float ndarray, optional, default: None
        Upper detection limit(s) for censored data. In multivariate cases the
        limits need to be defined as a vector; a scalar value is sufficient in
        a univariate case. If the data is not censored from above in a
        particular dimension, assign None to that position of the ndarray.
    alpha_lim: float, optional, default:None
        Introduces a lower limit to the probability density within the
        n-orthotope defined by the truncation limits. Assigning a reasonable
        minimum (such as 1e-4) can be useful when the mean of the distribution
        is several standard deviations from the truncation limits and the
        sample size is small. Such cases without a limit often converge to
        distant means with inflated variances. Besides being incorrect
        estimates, those solutions only offer negligible reduction in the
        negative log likelihood, while making subsequent sampling of the
        truncated normal distribution very challenging.

    Returns
    -------
    mu: float scalar or ndarray
        Mean of the fitted probability distribution. A vector of means is
        returned in a multivariate case.
    COV: float scalar or 2D ndarray
        Covariance matrix of the fitted probability distribution. A 2D square
        ndarray is returned in a multi-dimensional case, while a single
        variance (not standard deviation!) value is returned in a univariate
        case.

    """

    verbose = False
    if verbose:
        log_msg('------------ DETAILS OF MLE FITTING --------------')
        log_msg('Detection limits: [low; up]')
        for low, up in zip(det_lower, det_upper):
            log_msg('[{}; {}]'.format(low, up))

    # extract some basic information about the number of dimensions and the
    # number of samples from raw data
    samples = np.asarray(samples)
    if samples.ndim == 1:
        ndims = 1
        nsamples = len(samples)
        samplesT = samples
    else:
        ndims, nsamples = samples.shape
        samplesT = np.transpose(samples)
    if verbose:
        log_msg('')
        log_msg('Number of dimensions: {}'.format(ndims))
        log_msg('Number of samples: {}'.format(nsamples))

    mu_hat = np.mean(samplesT, axis=0)
    # replace zero standard dev with negligible standard dev
    if ndims == 1:
        sig_hat = np.maximum(1e-6, np.std(samplesT, axis=0))
    else:
        sig_hat = np.std(samplesT, axis=0)
        sig_zero_id = np.where(sig_hat == 0.0)[0]
        sig_hat[sig_zero_id] = 1e-6 * np.abs(mu_hat[sig_zero_id])

    if (det_lower is not None) and (det_upper is not None):
        det_upper_adj = (det_upper - mu_hat) / sig_hat
        det_lower_adj = (det_lower - mu_hat) / sig_hat

    if (tr_lower is not None) and (tr_upper is not None):
        tr_upper_adj = (tr_upper - mu_hat) / sig_hat
        tr_lower_adj = (tr_lower - mu_hat) / sig_hat

    if verbose:
        log_msg('')
        log_msg('Method of moments estimates: [mu, sig]')
        if ndims == 1:
            log_msg(f"[{mu_hat}, {sig_hat}]")
        else:
            for mu_i, sig_i in zip(mu_hat, sig_hat):
                log_msg(f"[{mu_i}, {sig_i}]")

    samplesT = (samplesT - mu_hat) / sig_hat

    mu_hatc = np.mean(samplesT, axis=0)
    sig_hatc = np.std(samplesT, axis=0)
    if ndims == 1:
        if sig_hatc == 0:
            sig_hatc = 0.999
    else:
        sig_zero_id = np.where(sig_hatc == 0.0)[0]
        sig_hatc[sig_zero_id] = 0.999

    if verbose:
        log_msg('')
        log_msg('Standardized estimates: [mu, sig]')
        for mu_i, sig_i in zip(mu_hatc, sig_hatc):
            log_msg('[{}, {}]'.format(mu_i, sig_i))

    # define initial values of distribution parameters using simple estimates
    if ndims == 1:
        if nsamples > ndims:
            # we perturb the mu to improve optimization
            mu_init = mu_hatc + sig_hatc*0.1
        else:
            mu_init = mu_hatc
        # use biased estimate for std, because MLE will converge to that anyway
        #sig_init = np.std(samples, ddof=0)
        sig_init = sig_hatc
        # replace zero variance with negligible variance
        if sig_init == 0.0:
            sig_init = 1e-6 * np.abs(mu_init)
        rho_init=()
        fit_rho = False
        # prepare a vector of initial values
        inits = np.asarray([mu_init, sig_init])
    else:
        if nsamples > ndims:
            # we perturb the mu to improve optimization
            mu_init = mu_hatc + sig_hatc*0.1
        else:
            mu_init = mu_hatc
        # use biased estimate, see comment above
        #sig_init = np.std(samples, axis=1, ddof=0)
        sig_init = sig_hatc
        # replace zero variance with negligible variance
        sig_zero_id = np.where(sig_init == 0.0)[0]
        sig_init[sig_zero_id] = 1e-6 * np.abs(mu_init[sig_zero_id])
        # try to create the correlation matrix
        rho_init = np.corrcoef(np.transpose(samplesT))
        # If there is not enough samples, or the samples are not from a
        # multivariate normal distribution, the rho values might be nan.
        # First of all, we let the user know about this with the warning below.
        if np.isnan(rho_init).any():
            show_warning(
                "Some elements of the correlation matrix are not valid. Make "
                "sure the provided raw EDP samples are appropriate to fit the "
                "specified type of distribution and all EDP dimensions are "
                "characterized by non-zero variance. The invalid correlation "
                "estimates are replaced with zero values.")

        # Second, we replace the nan values with zeros.
        rho_init = np.nan_to_num(rho_init)

        # Third, check if the resulting correlation matrix is positive
        # semidefinite. If not, then rather than trying to patch it, in this
        # case, we assume uncorrelated samples and make sure that at least the
        # mu and sig values will be OK.
        pos_sem_def = np.all(np.linalg.eigvals(rho_init) >= 0.)
        if not pos_sem_def:
            show_warning("The number of samples is not sufficiently large to estimate "
                "the correlation matrix. We assume uncorrelated EDPs.")
            rho_init = np.zeros(rho_init.shape)

        # The next line ensures that the diagonal is exactly 1.0. It is an
        # extra layer of safety that is almost always unnecessary.
        np.fill_diagonal(rho_init,1.0)

        fit_rho = False # Currently, rho is not estimated
        # TODO: introduce a method to get better estimates of rho

        if fit_rho:
            # collect the independent values (i.e. elements above the main
            # diagonal) from the correlation matrix in a list
            rho_init_ids = np.triu_indices(ndims, k=1)
            rho_init_list = rho_init[rho_init_ids]
            # save the ids of the elements below the main diagonal for future use
            rho_init_ids2 = rho_init_ids[::-1]

            # prepare a vector of initial values
            inits = np.concatenate([mu_init, sig_init, rho_init_list])

        else:

            # prepare a vector of initial values
            inits = np.concatenate([mu_init, sig_init])

    # If the distribution is censored or truncated, check if the number of
    # samples is greater than the number of unknowns. If not, show a warning.
    if fit_rho:
        if (((tr_lower is not None) or (tr_upper is not None)
            or (det_lower is not None) or (det_upper is not None))
           and (len(inits) >= nsamples)):
            if verbose: log_msg('samples:',nsamples,'unknowns:',len(inits))
            show_warning(
                "The number of samples is less than the number of unknowns. There "
                "is no unique solution available for such a case. Expect a poor "
                "estimate of the distribution (especially the covariance matrix). "
                "Either provide more samples, or relax the assumed dependencies "
                "between variables, or remove the truncation/detection limits to "
                "improve the situation."
            )

    # define the bounds for the distribution parameters
    # mu is not bounded
    #mu_bounds = [(-np.inf, np.inf) for t in range(ndims)]
    mu_bounds = [(-10.0, 10.0) for t in range(ndims)]
    # sig is bounded below at (0
    #sig_bounds = [(np.nextafter(0,1), np.inf) for s in range(ndims)]
    sig_bounds = [(0.05, 20.0) for s in range(ndims)]
    # rho is bounded on both sides by (-1,1)
    # Note that -1.0 and 1.0 are not allowed to avoid numerical problems due to
    # a singular and/or non-positive definite covariance matrix
    if ((ndims > 1) and (fit_rho)):
        rho_bounds = [(-1.+1e-3, 1.-1e-3) for r in range(len(rho_init_list))]
    else:
        # there is no need for rho bounds in a univariate case and when we do not fit rho
        rho_bounds = []
    # create a lower and an upper bounds vector
    #bounds = mu_bounds + sig_bounds + rho_bounds
    bounds = mu_bounds + sig_bounds + rho_bounds
    bnd_lower, bnd_upper = np.transpose(bounds)
    if verbose:
        log_msg('')
        log_msg('Bounds: [mu, sig]')
        for mu_i, sig_i in zip(bounds, sig_bounds):
            log_msg('[{}, {}]'.format(mu_i, sig_i))

        if fit_rho:
            log_msg(rho_bounds)

        log_msg('')
        log_msg('Initial values:')
        [log_msg(i) for i in inits]

    # create a convenience function that converts a vector of distribution
    # parameters to the standard mu and COV arrays
    def _get_mu_COV(params, rho, unbiased=False):
        """
        The unbiased flag controls if the bias in standard deviation
        estimates shall be corrected during conversion.
        """
        if ndims == 1:
            mu, COV = params
        else:
            mu = params[:ndims]
            sig = params[ndims:2*ndims]
            if unbiased:
                sig = sig * np.sqrt(nsamples / (nsamples-1))

            # reconstruct the covariance matrix
            COV = np.outer(sig, sig)
            if len(params) > 2*ndims:
                rho_list = params[2*ndims:]
                # add correlation estimates above...
                COV[rho_init_ids] = COV[rho_init_ids] * rho_list
                # and below the main diagonal
                COV[rho_init_ids2] = COV[rho_init_ids2] * rho_list
            else:
                COV = np.outer(sig, sig) * rho

        return mu, COV

    # create the negative log likelihood function for censored data from a
    # truncated multivariate normal distribution
    def _neg_log_likelihood(params, rho, enforce_bounds=False):

        verbose_NLL = False
        #if verbose_NLL: log_msg()
        params_to_show = params[:6]
        #params_to_show = np.sum(params) # this is useful when there are many

        # first, check if the parameters are within the pre-defined bounds
        if enforce_bounds:
            if ((params > bnd_lower) & (params < bnd_upper)).all(0) == False:
                # if they are not, then return an infinite value to discourage the
                # optimization algorithm from going in that direction
                if verbose_NLL: log_msg(params_to_show, 'out of bounds', 1e10)
                return 1e10

        # return inf if there is nan in params:
        if np.isnan(np.sum(params)):
            if verbose_NLL: log_msg(params_to_show, 'nan in params', 1e10)
            return 1e10

        # reconstruct the mu and COV arrays from the parameters
        mu, COV = _get_mu_COV(params, rho)
        sig = params[ndims:2*ndims]

        if ndims >= 2:
            pos_sem_def = np.all(np.linalg.eigvals(COV) >= 0.)
            #TODO: replace this with Cholesky decomposition check
            if not pos_sem_def:
                if verbose_NLL: log_msg(params_to_show, 'COV not pos sem def', 1e10)
                return 1e10

        # calculate the probability density within the truncation limits
        if (tr_lower is not None) and (tr_upper is not None):

            alpha, eps_alpha = mvn_orthotope_density(mu, COV,
                                                     tr_lower_adj,
                                                     tr_upper_adj)
            #if verbose: log_msg(tr_lower_adj, tr_upper_adj, mu, COV, alpha)
            # If the error in the alpha estimate is too large, then we are
            # beyond the applicability limits of the function used for
            # estimating alpha. Show a warning message and try to find another
            # solution by discouraging the optimization algorithm from going in
            # this direction.
            if alpha <= 100.*eps_alpha: #i.e. max. error is limited at 1%
                # Note: throwing an error here would be too extreme, because it
                # would stop the analysis completely, while a solution might be
                # reached if we let the optimization algorithm converge to it.
                if msg[0] == False:
                    warnings.warn(UserWarning(
                        'The density of the joint probability distribution '
                        'within the truncation limits is too small and '
                        'cannot be estimated with sufficiently high '
                        'accuracy.'
                    ))
                    msg[0] = True

                if verbose_NLL: log_msg(params_to_show, 'alpha estimate not applicable in truncs', 1e10)
                return 1e10

            # If a lower limit was prescribed for alpha, it should also be
            # enforced here
            if (alpha_lim is not None) and (alpha < alpha_lim):
                if msg[1] == False:
                    warnings.warn(UserWarning(
                        'The density of the joint probability distribution '
                        'within the truncation limits is less than the '
                        'prescribed minimum limit.'
                    ))
                    msg[1] = True

                if verbose_NLL: log_msg(params_to_show, 'not enough prob mass within truncs', 1e10)
                return 1e10

        else:
            alpha, eps_alpha = 1., 0.

        # calculate the likelihood for each available sample
        likelihoods = multivariate_normal.pdf(samplesT, mean=mu, cov=COV,
                                              allow_singular=True)

        # Zeros are a result of limited floating point precision. Replace them
        # with the smallest possible positive floating point number to
        # improve convergence.
        likelihoods = np.clip(likelihoods, a_min=np.nextafter(0,1), a_max=None)

        # calculate the likelihoods corresponding to censored data (if any)
        if censored_count > 0:

            # calculate the probability density within the detection limits
            det_alpha, eps_alpha = mvn_orthotope_density(mu, COV,
                                                         det_lower_adj,
                                                         det_upper_adj)
            # Similarly to alpha above, make sure that det_alpha is estimated
            # with sufficient accuracy.
            if det_alpha <= 100.*eps_alpha:
                if msg[2] == False:
                    warnings.warn(
                        'The density of the joint probability distribution '
                        'within the detection limits is too small and '
                        'cannot be estimated with sufficiently high '
                        'accuracy. '
                        '(alpha: '+str(det_alpha)+' eps: '+str(eps_alpha)+')'
                    )
                    msg[2] = True

                if verbose_NLL: log_msg(params_to_show, 'alpha estimate not applicable in dets', 1e10)
                return 1e10

            # calculate the likelihood of censoring a sample
            cen_likelihood = (alpha - det_alpha) / alpha

            # make sure that the likelihood is a positive number
            cen_likelihood = max(cen_likelihood, np.nextafter(0,1))

            if verbose_NLL :
                pass
                #log_msg('dets and cen_liks')
                #log_msg(det_lower, det_lower_adj)
                #log_msg(det_upper, det_upper_adj)
                #log_msg(det_alpha, cen_likelihood)

        else:
            # If the data is not censored, use 1.0 for cen_likelihood to get a
            # zero log-likelihood later. Note that although this is
            # theoretically not correct, it does not alter the solution and
            # it is numerically much more convenient than working around the
            # log of zero likelihood.
            cen_likelihood = 1.

        # calculate the total negative log-likelihood
        NLL = -(
            np.sum(np.log(likelihoods))                 # from samples
            - nsamples*np.log(alpha)                    # truncation influence
            + censored_count*np.log(cen_likelihood)     # censoring influence
        )

        # normalize the likelihoods with the sample count
        NLL = NLL/nsamples
        #log_msg(mu[-4:], NLL)
        #log_msg(np.sqrt(np.diagonal(COV))[-4:],NLL)

        if verbose_NLL: log_msg(f"{params_to_show} 'all good' {NLL}")
        #print(cen_likelihood)

        return NLL

    # initialize the message flags
    msg = [False, False, False]

    if verbose:
        log_msg('')
        log_msg('Initial NLL value: {}'.format(_neg_log_likelihood(inits, rho_init)))

    # perturbation
    #inits[:ndims] = inits[:ndims]+np.random.uniform(low=-0.5, high=0.5, size=ndims)
    #inits[ndims:2*ndims] += 0.5

    if verbose:
        t_0 = time.time()
    # minimize the negative log-likelihood function
    #out = minimize(_neg_log_likelihood, inits, args=(rho_init, True),
    #               bounds=bounds, method='TNC')

    # Global optimization with a more sophisticated method is only
    # reasonable if we have a sufficiently large number of samples.
    # Considering the size of the covariance matrix, we are looking for at least
    # ndims^2 samples to use differential evolution.
    # This is turned off for now, will come back later as an optional method
    # that the user can control directly
    if False:
    #if nsamples > 2.0 * ndims**2.0:

        out_d = differential_evolution(_neg_log_likelihood, mu_bounds + sig_bounds,
                                       args=(rho_init,),
                                       maxiter=200,
                                       polish=False)
        mu_sig_vals = out_d.x

        if verbose:
            log_msg(out_d)
            # log_msg(out.fun, out.nfev, out.nit, out.message, out.x)
            log_msg('runtime: ', time.time() - t_0)
    else:
        mu_sig_vals = np.array(inits[:2*ndims])

    # Minimize the negative log-likelihood function using the adaptive
    # Adaptive Nelder-Mead algorithm (Gao and Han, 2012)
    if nsamples > ndims:
        out_m = minimize(_neg_log_likelihood,
                         np.concatenate([mu_sig_vals, inits[2*ndims:]]),
                       args=(rho_init,True), method='Nelder-Mead',
                       options=dict(maxiter=500,
                                    maxfev=2000,
                                    xatol = 0.01,
                                    fatol = 1e-10,
                                    adaptive=True)
                       )
        #print(out_m.nfev, out_m.nit)
        if verbose:
            log_msg('')
            log_msg('Nelder-Mead minimization results:')
            log_msg('\t{}'.format(out_m.message))
            log_msg('\tfun: {}'.format(out_m.fun))
            log_msg('runtime: {}'.format(time.time() - t_0))

        # reconstruct the mu and COV arrays from the solutions and return them
        mu, COV = _get_mu_COV(out_m.x, rho_init, unbiased=True)
    else:
        mu, COV = _get_mu_COV(mu_sig_vals, rho_init, unbiased=True)

    if verbose:
        if ndims >= 2:
            log_msg('mu vs mu_init')
            show_matrix(list(zip(mu,mu_init)))
            sig = np.sqrt(np.diagonal(COV))
            log_msg('sig vs sig_init')
            show_matrix(list(zip(sig,sig_init)))
            rho = (COV/np.outer(sig,sig))[-8:,-8:]
            log_msg('rho')
            show_matrix(rho)
            log_msg('rho_init')
            show_matrix(rho_init[-8:,-8:])
        else:
            log_msg('mu vs mu_init')
            log_msg(mu, mu_init)
            sig = np.sqrt(COV)
            log_msg('sig vs sig_init')
            log_msg(sig, sig_init)

    if ndims >= 2:
        sig = np.sqrt(np.diagonal(COV))
        rho = (COV / np.outer(sig, sig))
        sig = sig * sig_hat
        COV = np.outer(sig, sig) * rho
    else:
        sig = np.sqrt(COV)
        sig = sig*sig_hat
        COV = sig**2.0
    mu = mu_hat + mu * sig_hat

    if verbose:
        log_msg('')
        log_msg('Final values: [mu, sig]')
        for mu_i, sig_i in zip(mu, sig):
            log_msg('[{}, {}]'.format(mu_i, sig_i))

        #log_msg(COV)

    return mu, COV


class RandomVariable(object):
    """
    Characterizes a Random Variable (RV) that represents a source of
    uncertainty in the calculation.

    The uncertainty can be described either through raw data or through a
    pre-defined distribution function. When using raw data, provide potentially
    correlated raw samples in a 2 dimensional array. If the data is left or
    right censored in any number of its dimensions, provide the list of
    detection limits and the number of censored samples. No other information
    is needed to define the object from raw data. Then, either resample the raw
    data, or fit a prescribed distribution to the samples and sample from that
    distribution later.

    Alternatively, one can choose to prescribe a distribution type and its
    parameters and sample from that distribution later.

    Parameters
    ----------
    ID: int
    dimension_tags: str array
        A series of strings that identify the stochastic model parameters that
        correspond to each dimension of the random variable. When the RV is one
        dimensional, the dim_tag is a single string. In multi-dimensional
        cases, the order of strings shall match the order of elements provided
        as other inputs.
    raw_data: float scalar or ndarray, optional, default: None
        Samples of an uncertain variable. The samples can describe a
        multi-dimensional random variable if they are arranged in a 2D ndarray.
    detection_limits: float ndarray, optional, default: None
        Defines the limits for censored data. The limits need to be defined in
        a 2D ndarray that is structured as two vectors with N elements. The
        vectors collect left and right limits for the N dimensions. If the data
        is not censored in a particular direction, assign None to that position
        of the ndarray. Replacing one of the vectors with None will assign no
        censoring to all dimensions in that direction. The default value
        corresponds to no censoring in either dimension.
    censored_count: int, optional, default: None
        The number of censored samples that are beyond the detection limits.
        All samples outside the detection limits are aggregated into one set.
        This works the same way in one and in multiple dimensions. Prescription
        of censored sample counts for sub-regions of the input space outside
        the detection limits is not yet supported. If such an approach is
        desired, the censored raw data shall be used to fit a distribution in a
        pre-processing step and the fitted distribution can be specified for
        this random variable.
    distribution_kind: {'normal', 'lognormal', 'multinomial'}, optional, default: None
        Defines the type of probability distribution when raw data is not
        provided, but the distribution is directly specified. When part of the
        data is normal in log space, while the other part is normal in linear
        space, define a list of distribution tags such as ['normal', 'normal',
        'lognormal']. Make sure that the covariance matrix is based on log
        transformed data for the lognormally distributed variables! Mixing
        normal distributions with multinomials is not supported.
    theta: float scalar or ndarray, optional, default: None
        Median of the probability distribution. A vector of medians is expected
        in a multi-dimensional case.
    COV: float scalar or 2D ndarray, optional, default: None
        Covariance matrix of the random variable. In a multi-dimensional case
        this parameter has to be a 2D square ndarray, and the number of its
        rows has to be equal to the number of elements in the supplied theta
        vector. In a one-dimensional case, a single value is expected that
        equals the variance (not the standard deviation!) of the distribution.
        The COV for lognormal variables is assumed to be specified in
        logarithmic space.
    corr_ref: {'pre', 'post'}, optional, default: 'pre'
        Determines whether the correlations prescribed by the covariance matrix
        refer to the distribution functions before or after truncation. The
        default 'pre' setting assumes that pre-truncation correlations are
        prescribed and creates a multivariate normal distribution using the
        COV matrix. That distribution is truncated according to the prescribed
        truncation limits. The other option assumes that post-truncation
        correlations are prescribed. The post-truncation distribution
        is not multivariate normal in general. Currently we use a Gaussian
        copula to describe the dependence between the truncated variables.
        Similarly to other characteristics, the `corr_ref` can be defined as a
        single string, or a vector of strings. The former assigns the same
        option to all dimensions, while the latter allows for more flexible
        assignment by setting the corr_ref for each dimension individually.
    p_set: float vector, optional, default: None
        Probabilities of a finite set of events described by a multinomial
        distribution. The RV will have binomial distribution if only one
        element is provided in this vector. The number of events equals the
        number of vector elements if their probabilities sum up to 1.0. If the
        sum is less than 1.0, then an additional event is assumed with the
        remaining probability of occurrence assigned to it. The sum of
        event probabilities shall never be more than 1.0.
    truncation_limits: float ndarray, optional, default: None
        Defines the limits for truncated distributions. The limits need to be
        defined in a 2D ndarray that is structured as two vectors with N
        elements. The vectors collect left and right limits for the N
        dimensions. If the distribution is not truncated in a particular
        direction, assign None to that position of the ndarray. Replacing one
        of the vectors with None will assign no truncation to all dimensions
        in that direction. The default value corresponds to no truncation in
        either dimension.
    """

    def __init__(self, ID, dimension_tags,
                 raw_data=None, detection_limits=None, censored_count=None,
                 distribution_kind=None,
                 theta=None, COV=None, corr_ref='pre', p_set=None,
                 truncation_limits=None):
        self._ID = ID

        self._dimension_tags = np.asarray(dimension_tags)

        if raw_data is not None:
            raw_data = np.atleast_1d(raw_data)
            if len(raw_data.shape) > 1:
                self._ndim, self._ncount = raw_data.shape[:2]
            else:
                self._ndim = 1
                self._ncount = raw_data.shape[0]
        self._raw_data = raw_data

        if self._raw_data is not None:
            self._detection_limits = self._convert_limits(detection_limits)
            self._censored_count = censored_count
        else:
            self._detection_limits = self._convert_limits(None)
            self._censored_count = None

        if distribution_kind is not None:
            distribution_kind = np.asarray(distribution_kind)
        self._distribution_kind = distribution_kind

        if self._distribution_kind is not None:
            if theta is not None:
                theta = np.asarray(theta)
                if theta.shape == ():
                    self._ndim = 1
                else:
                    self._ndim = theta.shape[0]
            self._theta = theta

            if COV is not None:
                COV = np.asarray(COV)
            self._COV = COV

            self._corr_ref = np.asarray(corr_ref)

            if p_set is not None:
                p_set = np.asarray(p_set)
                if np.sum(p_set) < 1.:
                    p_set = np.append(p_set, 1.-np.sum(p_set))
                self._ndim = 1
            self._p_set = p_set

            tr_limits = self._convert_limits(truncation_limits)
            self._tr_limits_pre, self._tr_limits_post = \
                self._create_pre_post_tr_limits(tr_limits)
        else:
            self._theta = None
            self._COV = None
            self._corr_ref = corr_ref
            self._p_set = None
            self._tr_limits_pre = self._convert_limits(None)
            self._tr_limits_post = deepcopy(self._tr_limits_pre)

        # perform some basic checks to make sure that the provided data will be
        # sufficient to define a random variable
        if self._raw_data is None:

            # if the RV is defined by providing distribution data...
            if self._distribution_kind is None:
                raise ValueError(
                    "Either raw samples or a distribution needs to be defined "
                    "for a random variable."
                )

            if ((self._distribution_kind.shape!=())
                or ((self._distribution_kind.shape==())
                    and (self._distribution_kind in ['normal', 'lognormal']))):
                if (self._theta is None) or (self._COV is None):
                    raise ValueError(
                        "Normal and lognormal distributions require theta and "
                        "COV parameters."
                    )

            if ((self._distribution_kind.shape==())
                and (self._distribution_kind in ['multinomial'])
                and (self._p_set is None)):
                raise ValueError(
                    "Multinomial distributions require a set of p values as "
                    "parameters."
                )
        else:

            # if the RV is defined through raw samples
            if ((self._detection_limits is None) !=
                (self._censored_count is None)):
                raise ValueError(
                    "Definition of censored data requires information about "
                    "the detection limits and the number of censored samples."
                )

    def _convert_limits(self, limits):
        """
        Convert None values to infinites in truncation and detection limits.

        """
        if hasattr(self, '_ndim') and (limits is not None):
            # convert single-element limits array into a nested
            # structure that is easier to work with
            if not isinstance(limits[0], (list, tuple, np.ndarray)):
                limits[0] = [limits[0], ]
            if not isinstance(limits[1], (list, tuple, np.ndarray)):
                limits[1] = [limits[1], ]

            # assign a vector of None in place of a single None value
            if (len(limits[0]) == 1) and (limits[0][0] is None):
                limits[0] = [None for d in range(self._ndim)]
            if (len(limits[1]) == 1) and (limits[1][0] is None):
                limits[1] = [None for d in range(self._ndim)]

            # replace None values with infinite limits
            for l_i, l in enumerate(limits[0]):
                if l is None:
                    limits[0][l_i] = -np.inf
            for l_i, l in enumerate(limits[1]):
                if l is None:
                    limits[1][l_i] = np.inf

            limits = np.array(limits, dtype=np.float64)

        return limits

    def _create_pre_post_tr_limits(self, truncation_limits):
        """
        Separates the truncation limits into two groups: (i) `pre` truncation
        limits apply to distributions where the correlations refer to the
        joint distribution before truncation; (ii) `post` truncation limits
        apply to distributions after truncation. Truncation in the latter case
        is applied differently, hence the need to separate the two types of
        modifications.

        """
        if (truncation_limits is not None) and (hasattr(self, '_ndim')):
            tr_lower, tr_upper = truncation_limits
            CR = self._corr_ref

            # a single value or identical values means one setting
            # applies to all dims
            if (CR.size == 1) or (np.unique(CR).size == 1):
                if CR.size > 1:
                    CR = CR[0]
                if CR == 'pre':
                    trl_pre = deepcopy(truncation_limits)
                    trl_post = None
                elif CR == 'post':
                    trl_pre = None
                    trl_post = deepcopy(truncation_limits)
            else:
                # otherwise assign the appropriate limits to each dim
                tr_lower_pre, tr_lower_post = -np.ones((2,self._ndim))*np.inf
                tr_upper_pre, tr_upper_post = np.ones((2,self._ndim))*np.inf
                tr_lower_pre[CR == 'pre'] = tr_lower[CR == 'pre']
                tr_upper_pre[CR == 'pre'] = tr_upper[CR == 'pre']
                tr_lower_post[CR == 'post'] = tr_lower[CR == 'post']
                tr_upper_post[CR == 'post'] = tr_upper[CR == 'post']

                trl_pre = np.asarray([tr_lower_pre, tr_upper_pre])
                trl_post = np.asarray([tr_lower_post, tr_upper_post])
        else:
            trl_pre, trl_post = None, None

        return trl_pre, trl_post

    def _move_to_log(self, raw_data, distribution_list):
        """
        Convert data to log space for the lognormal variables.

        """

        if distribution_list is None:
            data = raw_data
        elif distribution_list.shape == ():
            # meaning identical distribution families
            data = raw_data
            if distribution_list == 'lognormal':
                if np.min(data) > 0.:
                    data = np.log(raw_data)
                else:
                    # this can gracefully handle non-positive limits for
                    # lognormal distributions
                    min_float = np.nextafter(0, 1)
                    data = np.log(np.clip(raw_data, a_min=min_float,
                                          a_max=None))
                    if np.asarray(data).shape == ():
                        if data == np.log(min_float):
                            data = -np.inf
                    # Although the following code would help with other
                    # incorrect data, it might also hide such problems and
                    # the current implementation does not need it, so it
                    # is disabled for now.
                    #else:
                    #    data[data==np.log(min_float)] = -np.inf
        else:
            data = deepcopy(raw_data)
            for dim, dk in enumerate(distribution_list):
                if dk == 'lognormal':
                    if np.min(data[dim]) > 0.:
                        data[dim] = np.log(data[dim])
                    else:
                        # this can gracefully handle non-positive limits for
                        # lognormal distributions
                        min_float = np.nextafter(0, 1)
                        data[dim] = np.log(np.clip(data[dim], a_min=min_float,
                                                   a_max=None))
                        if np.asarray(data[dim]).shape == ():
                            if data[dim] == np.log(min_float):
                                data[dim] = -np.inf
                        #else:
                        #    data[dim][data[dim]==np.log(min_float)] = -np.inf

        return data

    def _return_from_log(self, raw_data, distribution_list):
        """
        Convert data back to linear space for the lognormal variables.

        """
        if distribution_list.shape == ():
            # meaning identical distribution families
            data = raw_data
            if distribution_list == 'lognormal':
                data = np.exp(raw_data)
        elif distribution_list.shape == (1,):
            if distribution_list[0] == 'lognormal':
                data = np.exp(raw_data)
            else:
                data = raw_data
        else:
            data = raw_data
            for dim, dk in enumerate(distribution_list):
                if dk == 'lognormal':
                    data[dim] = np.exp(data[dim])

        return data

    @property
    def distribution_kind(self):
        """
        Return the assigned probability distribution family.
        """
        return self._distribution_kind

    @property
    def theta(self):
        """
        Return the median value(s) of the probability distribution.
        """
        if self._theta is not None:
            return self._theta
        else:
            raise ValueError(
                "The median of the probability distribution of this random "
                "variable is not yet specified."
            )

    @property
    def mu(self):
        """
        Return the mean value(s) of the probability distribution. Note that
        the mean value is in log space for lognormal distributions.

        """
        return self._move_to_log(self.theta, self._distribution_kind)

    @property
    def COV(self):
        """
        Return the covariance matrix of the probability distribution. Note that
        the covariances are in log space for lognormal distributions.
        """
        if self._COV is not None:
            return self._COV
        else:
            raise ValueError(
                "The covariance matrix of the probability distribution of "
                "this random variable is not yet specified."
            )

    @property
    def corr(self):
        """
        Return the correlation matrix of the probability distribution. Note that
        correlation coefficient correspond to the joint distribution in log
        space for lognormal distributions.
        """
        if self._COV is not None:
            return self._COV / np.outer(self.sig, self.sig)
        else:
            raise ValueError(
                "The correlation matrix of the probability distribution of "
                "this random variable is not yet specified."
            )

    @property
    def var(self):
        """
        Return the variance vector of the probability distribution. Note that
        the variances are in log space for lognormal distributions.
        """
        if self._COV is not None:
            return np.diagonal(self._COV)
        else:
            raise ValueError(
                "The covariance matrix of the probability distribution of "
                "this random variable is not yet specified."
            )

    @property
    def sig(self):
        """
        Return the standard deviation vector of the probability distribution.
        Note that the standard deviations are in log space for lognormal
        distributions.
        """
        if self._COV is not None:
            return np.sqrt(self.var)
        else:
            raise ValueError(
                "The covariance matrix of the probability distribution of "
                "this random variable is not yet specified."
            )

    @property
    def dimension_tags(self):
        """
        Return the tags corresponding to the dimensions of the variable.

        """
        # this is very simple for now
        return self._dimension_tags

    @property
    def detection_limits(self):
        """
        Return the detection limits corresponding to the raw data in linear
        space.

        """
        # this is very simple for now
        return self._detection_limits

    @property
    def det_lower(self):
        """
        Return the lower detection limit(s) corresponding to the raw data in
        either linear or log space according to the distribution.

        """
        if self._detection_limits is None:
            return None
        else:
            return self._move_to_log(self._detection_limits[0],
                                     self._distribution_kind)

    @property
    def det_upper(self):
        """
        Return the upper detection limit(s) corresponding to the raw data in
        either linear or log space according to the distribution.

        """
        if self._detection_limits is None:
            return None
        else:
            return self._move_to_log(self._detection_limits[1],
                                     self._distribution_kind)

    @property
    def tr_limits_pre(self):
        """
        Return the `pre` truncation limits of the probability distribution in
        linear space.

        """
        # this is very simple for now
        return self._tr_limits_pre

    @property
    def tr_limits_post(self):
        """
        Return the `post` truncation limits of the probability distribution in
        linear space.

        """
        # this is very simple for now
        return self._tr_limits_post

    @property
    def tr_lower_pre(self):
        """
        Return the lower `pre` truncation limit(s) corresponding to the
        distribution in either linear or log space according to the
        distribution.

        """
        if self._tr_limits_pre is None:
            return None
        else:
            return self._move_to_log(self._tr_limits_pre[0],
                                     self._distribution_kind)

    @property
    def tr_upper_pre(self):
        """
        Return the upper `pre` truncation limit(s) corresponding to the
        distribution in either linear or log space according to the
        distribution.

        """
        if self._tr_limits_pre is None:
            return None
        else:
            return self._move_to_log(self._tr_limits_pre[1],
                                     self._distribution_kind)

    @property
    def tr_lower_post(self):
        """
        Return the lower `post` truncation limit(s) corresponding to the
        distribution in either linear or log space according to the
        distribution.

        """
        if self._tr_limits_post is None:
            return None
        else:
            return self._move_to_log(self._tr_limits_post[0],
                                     self._distribution_kind)

    @property
    def tr_upper_post(self):
        """
        Return the upper `post` truncation limit(s) corresponding to the
        distribution in either linear or log space according to the
        distribution.

        """
        if self._tr_limits_post is None:
            return None
        else:
            return self._move_to_log(self._tr_limits_post[1],
                                     self._distribution_kind)

    @property
    def censored_count(self):
        """
        Return the number of samples beyond the detection limits.

        """
        if self._censored_count is None:
            return 0
        else:
            return self._censored_count

    @property
    def samples(self):
        """
        Return the pre-generated samples from the distribution.

        """
        if hasattr(self, '_samples'):
            return self._samples
        else:
            return None

    @property
    def raw(self):
        """
        Return the pre-assigned raw data.

        """
        if hasattr(self, '_raw_data'):
            return self._raw_data
        else:
            return None

    def fit_distribution(self, distribution_kind, truncation_limits=None):
        """
        Estimate the parameters of a probability distribution from raw data.

        Parameter estimates are calculated using maximum likelihood estimation.
        If the data spans multiple dimensions, the estimates will also describe
        a multi-dimensional distribution automatically. Data censoring is also
        automatically taken into consideration following the detection limits
        specified previously for the random variable. Truncated target
        distributions can be specified through the truncation limits. The
        specified truncation limits are applied after the correlations are set.
        In other words, the corr_ref proprety of the RV is set to 'pre' when
        fitting a distribution.

        Besides returning the parameter estimates, their values are also stored
        as theta and COV attributes of the RandomVariable object for future
        use.

        Parameters
        ----------
        distribution_kind: {'normal', 'lognormal'} or a list of those
            Specifies the type of the probability distribution that is fit to
            the raw data. When part of the data is normal in log space, while
            the other part is normal in linear space, define a list of
            distribution tags such as ['normal', 'normal', 'lognormal'].
        truncation_limits: float ndarray, optional, default: None
            Defines the limits for truncated distributions. The limits need to
            be defined in a 2D ndarray that is structured as two vectors with N
            elements. The vectors collect left and right limits for the N
            dimensions. If the distribution is not truncated in a particular
            direction, assign None to that position of the ndarray. Replacing
            one of the vectors with None will assign no truncation to all
            dimensions in that direction. The default value corresponds to no
            truncation in either dimension.

        Returns
        -------
        theta: float scalar or ndarray
            Median of the probability distribution. A vector of medians is
            returned in a multi-dimensional case.
        COV: float scalar or 2D ndarray
            Covariance matrix of the probability distribution. A 2D square
            ndarray is returned in a multi-dimensional case.
        """

        # lognormal distribution parameters are estimated by fitting a normal
        # distribution to the data in log space
        distribution_kind = np.asarray(distribution_kind)
        data = self._move_to_log(self._raw_data, distribution_kind)

        # prepare the information on truncation
        if truncation_limits is not None:
            tr_lower, tr_upper = self._convert_limits(truncation_limits)
            tr_lower = self._move_to_log(tr_lower, distribution_kind)
            tr_upper = self._move_to_log(tr_upper, distribution_kind)
        else:
            tr_lower, tr_upper = None, None

        # convert the detection limits to log if needed
        if self.detection_limits is not None:
            det_lower, det_upper = self.detection_limits
            det_lower = self._move_to_log(det_lower, distribution_kind)
            det_upper = self._move_to_log(det_upper, distribution_kind)
        else:
            det_lower, det_upper = None, None

        # perform the parameter estimation
        mu, COV = tmvn_MLE(data,
                           tr_lower = tr_lower, tr_upper=tr_upper,
                           censored_count=self.censored_count,
                           det_lower=det_lower, det_upper=det_upper)

        # convert mu to theta
        theta = self._return_from_log(mu, distribution_kind)

        # store and return the parameters
        self._theta = theta
        self._COV = COV
        self._corr_ref = 'pre'
        #TODO: implement 'post' corr_ref as an option for fitting

        # store the distribution properties
        self._distribution_kind = distribution_kind
        self._tr_limits_pre = truncation_limits

        return theta, COV

    def sample_distribution(self, sample_size, preserve_order=False):
        """
        Sample the probability distribution assigned to the random variable.

        Normal distributions (including truncated and/or multivariate normal
        and lognormal) are sampled using the tmvn_rvs() method in this module.
        If post-truncation correlations are set for a dimension, the
        corresponding truncations are enforced after sampling by first applying
        probability integral transformation to transform samples from the
        non-truncated normal to standard uniform distribution, and then
        applying inverse probability integral transformation to transform the
        samples from standard uniform to the desired truncated normal
        distribution. Multinomial distributions are sampled using the
        multinomial method in scipy. The samples are returned and also stored
        in the `sample` attribute of the RV.

        If the random variable is defined by raw data only, we sample from the
        raw data.

        Parameters
        ----------
        sample_size: int
            Number of samples requested.
        preserve_order: bool, default: False
            Influences sampling from raw data. If True, the samples are copies
            of the first n rows of the raw data where n is the sample_size. This
            only works for sample_size <= raw data size. If False, the samples
            are drawn from the raw data pool with replacement.

        Returns
        -------
        samples: DataFrame
            Samples generated from the distribution. Columns correspond to the
            dimension tags that identify the variables.
        """

        if (not preserve_order) and (self._distribution_kind is not None):
            if ((self._distribution_kind.shape == ()) and
                (self._distribution_kind == 'multinomial')):

                # sampling the multinomial distribution
                samples = multinomial.rvs(1, self._p_set, size=sample_size)

                # convert the 2D sample array into a vector of integers
                outcomes = np.array([np.arange(len(self._p_set))])
                samples = np.matmul(samples, outcomes.T).flatten()
                samples = pd.DataFrame(np.transpose(samples),
                                       columns=self._dimension_tags)
            else:

                # sampling the truncated multivariate normal distribution
                raw_samples = tmvn_rvs(mu=self.mu, COV=self.COV,
                                       lower=self.tr_lower_pre,
                                       upper=self.tr_upper_pre,
                                       size=sample_size)

                raw_samples = np.transpose(raw_samples)

                # enforce post-truncation correlations if needed
                if self.tr_limits_post is not None:

                    lower, upper = self.tr_lower_post, self.tr_upper_post
                    for dim in range(self._ndim):
                        if (lower[dim] > -np.inf) or (upper[dim]<np.inf):

                            mu = self.mu[dim]
                            sig = np.sqrt(self.COV[dim,dim])

                            samples_U = norm.cdf(raw_samples[dim],loc=mu,scale=sig)

                            p_a = norm.cdf((lower[dim]-mu)/sig)
                            p_b = norm.cdf((upper[dim]-mu)/sig)

                            samples_U = samples_U * (p_b - p_a) + p_a

                            raw_samples[dim] = norm.ppf(samples_U, loc=mu, scale=sig)

                            # truncnorm has been 100x slower since scipy 1.4.1
                            # so I replaced it with the above solution.
                            #raw_samples[dim] = truncnorm.ppf(
                            #    samples_U, loc=mu, scale=sig,
                            #    a = (lower[dim]-mu)/sig, b=(upper[dim]-mu)/sig)

                # transform samples back from log space if needed
                samples = self._return_from_log(raw_samples,
                                                self._distribution_kind)

                samples = pd.DataFrame(data=np.transpose(np.atleast_2d(samples)),
                                       index=np.arange(sample_size),
                                       columns=self._dimension_tags)

                samples = samples.astype(np.float64)
        else:
            if self._raw_data is not None:

                if preserve_order:
                    if self._ncount >= sample_size:
                        if self._ndim > 1:
                            samples = self._raw_data[:, :sample_size]
                        else:
                            samples = self._raw_data[:sample_size]

                    else:
                        raise ValueError(
                            "The number of samples requested is larger than "
                            "number of raw data points available. Either "
                            "sample without preserving order or reduce the "
                            "sample size.")

                else:
                    # generate a random list of indices
                    id_list = np.random.randint(0, self._ncount, size=sample_size)

                    # get the raw data that corresponds to the random ids
                    if self._ndim > 1:
                        samples = self._raw_data[:, id_list]
                    else:
                        samples = self._raw_data[0, id_list]

                # put the samples in a DataFrame
                samples = pd.DataFrame(data=np.transpose(samples),
                                       index=np.arange(sample_size),
                                       columns=self._dimension_tags)

                samples = samples.astype(np.float64)

            else:
                raise ValueError(
                    "Either raw samples or a distribution needs to be defined "
                    "to sample a random variable.")

        self._samples = samples

        return self._samples

    def orthotope_density(self, lower=None, upper=None):
        """
        Estimate the probability density within an orthotope for a TMVN distr.

        Use the mvn_orthotope_density function in this module for the
        calculation. Pre-defined truncation limits for the RV are automatically
        taken into consideration. Limits for lognormal distributions shall be
        provided in linear space - the conversion is performed by the algorithm
        automatically. Pre- and post-truncation correlation is also considered
        automatically.

        Parameters
        ----------
        lower: float vector, optional, default: None
            Lower bound(s) of the orthotope. A scalar value can be used for a
            univariate RV; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from below in any dimension, use
            either 'None' or assign an infinite value (i.e. -numpy.inf) to
            that dimension.
        upper: float vector, optional, default: None
            Upper bound(s) of the orthotope. A scalar value can be used for a
            univariate RV; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from above in any dimension, use
            either 'None' or assign an infinite value (i.e. numpy.inf) to
            that dimension.

        Returns
        -------
        alpha: float
            Estimate of the probability density within the orthotope.
        eps_alpha: float
            Estimate of the error in alpha.

        """
        # get the orthotope density within the truncation limits
        if (self.tr_lower_pre is None) and (self.tr_upper_pre is None):
            alpha_0 = 1.
        else:
            alpha_0, __ = mvn_orthotope_density(self.mu, self.COV,
                                                self.tr_lower_pre, self.tr_upper_pre)

        # merge the specified limits with the pre-defined truncation limits
        lower, upper = self._convert_limits([lower, upper])
        lower = self._move_to_log(lower, self._distribution_kind)
        upper = self._move_to_log(upper, self._distribution_kind)

        # if there are post-truncation correlations defined, transform the
        # prescribed limits to 'pre' type limits
        if self.tr_limits_post is not None:
            lower_lim_post, upper_lim_post = (self.tr_lower_post,
                                              self.tr_upper_post)
            for dim in range(self._ndim):
                if ((lower_lim_post[dim] < lower[dim])
                     or (upper_lim_post[dim] > upper[dim])):
                    mu =self.mu[dim]
                    sig = np.sqrt(self.COV[dim, dim])
                    lim_U = truncnorm.cdf([lower[dim], upper[dim]],
                                          loc=mu, scale=sig,
                                          a=(lower_lim_post[dim]-mu)/sig,
                                          b=(upper_lim_post[dim]-mu)/sig)
                    lim_pre = norm.ppf(lim_U, loc=mu, scale=sig)
                    lower[dim], upper[dim] = lim_pre

        if self.tr_limits_pre is not None:
            lower_lim_pre, upper_lim_pre = (self.tr_lower_pre,
                                            self.tr_upper_pre)
            lower_lim_pre = np.maximum(lower_lim_pre, lower)
            upper_lim_pre = np.minimum(upper_lim_pre, upper)
        else:
            lower_lim_pre = lower
            upper_lim_pre = upper

        # get the orthotope density within the prescribed limits
        alpha, eps_alpha = mvn_orthotope_density(self.mu, self.COV,
                                                 lower_lim_pre, upper_lim_pre)

        # note that here we assume that the error in alpha_0 is negligible
        return min(alpha / alpha_0, 1.), eps_alpha / alpha_0


class RandomVariableSubset(object):
    """
    Provides convenient access to a subset of components of a RandomVariable.

    This object is useful when working with multivariate RVs, but it is used in
    all cases to provide a general approach.

    Parameters
    ----------
    RV: RandomVariable
        The potentially multivariate random variable that is accessed through
        this object.
    tags: str or list of str
        A string or list of strings that identify the subset of component we
        are interested in. These strings shall be among the `dimension_tags` of
        the RV.
    """

    def __init__(self, RV, tags):

        self._RV = RV
        self._tags = tags

    @property
    def tags(self):
        """
        Return the tags corresponding to the components in the RV subset.

        """
        # this is very simple for now
        return self._tags

    @property
    def samples(self):
        """
        Return the pre-generated samples of the selected component from the
        RV distribution.

        """
        samples = self._RV.samples

        if samples is not None:
            return samples[self._tags]
        else:
            return None

    def sample_distribution(self, sample_size, preserve_order=False):
        """
        Sample the probability distribution assigned to the connected RV.

        Note that this function will sample the full, potentially multivariate,
        distribution.

        Parameters
        ----------
        sample_size: int
            Number of samples requested.
        preserve_order: bool, default: False
            Influences sampling from raw data. If True, the samples are copies
            of the first n rows of the raw data where n is the sample_size. This
            only works for sample_size <= raw data size. If False, the samples
            are drawn from the raw data pool with replacement.

        Returns
        -------
        samples: DataFrame
            Samples of the selected component generated from the distribution.

        """
        samples = self._RV.sample_distribution(sample_size, preserve_order)

        return samples[self._tags]

    def orthotope_density(self, lower=None, upper=None):
        """
        Return the density within the orthotope in the marginal pdf of the RVS.

        The function considers the influence of every dependent variable in the
        RV on the marginal pdf of the RVS. Note that such influence only occurs
        when the RV is a truncated distribution and at least two variables are
        dependent. Pre- and post-truncation correlation is considered
        automatically.

        Parameters
        ----------
        lower: float vector, optional, default: None
            Lower bound(s) of the orthotope. A scalar value can be used for a
            univariate RVS; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from below in any dimension, use
            either 'None' or assign an infinite value (i.e. -numpy.inf) to
            that dimension.
        upper: float vector, optional, default: None
            Upper bound(s) of the orthotope. A scalar value can be used for a
            univariate RVS; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from above in any dimension, use
            either 'None' or assign an infinite value (i.e. numpy.inf) to
            that dimension.

        Returns
        -------
        alpha: float
            Estimate of the probability density within the orthotope.
        eps_alpha: float
            Estimate of the error in alpha.
        """

        # get the dimension tags from the parent RV and find the ones that
        # define this RVS
        dtags = self._RV.dimension_tags
        if dtags.size == 1:
            dtags = [dtags]
        sorter = np.argsort(dtags)
        tag_ids = sorter[np.searchsorted(dtags, self._tags, sorter=sorter)]

        # prepare the limit vectors and assign the limits to the appropriate
        # dimensions
        lower_full = [None for i in range(len(dtags))]
        upper_full = deepcopy(lower_full)

        if lower is not None:
            lower_full = np.asarray(lower_full)
            lower_full[tag_ids] = np.asarray(lower)
            lower_full = lower_full.tolist()

        if upper is not None:
            upper_full = np.asarray(upper_full)
            upper_full[tag_ids] = np.asarray(upper)
            upper_full = upper_full.tolist()

        # get the alpha value from the parent RV
        return self._RV.orthotope_density(lower_full, upper_full)
