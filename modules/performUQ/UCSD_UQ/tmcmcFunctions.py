"""authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, and Prof. J.P. Conte
affiliation: University of California, San Diego

"""  # noqa: INP001, D205, D400

import numpy as np
from runFEM import runFEM
from scipy.special import logsumexp


def initial_population(N, p):  # noqa: N803, D103
    IniPop = np.zeros((N, len(p)))  # noqa: N806
    for i in range(len(p)):
        IniPop[:, i] = p[i].generate_rns(N)
    return IniPop


def log_prior(s, p):  # noqa: D103
    logP = 0  # noqa: N806
    for i in range(len(s)):
        logP = logP + p[i].log_pdf_eval(s[i])  # noqa: N806
    return logP


def propose(current, covariance, n):  # noqa: D103
    return np.random.multivariate_normal(current, covariance, n)


def compute_beta(beta, likelihoods, prev_ESS, threshold):  # noqa: N803, D103
    old_beta = beta
    min_beta = beta
    max_beta = 2.0
    # rN = int(len(likelihoods) * 0.95)   #pymc3 uses 0.5
    rN = threshold * prev_ESS  # purdue prof uses 0.95  # noqa: N806
    new_beta = beta
    while max_beta - min_beta > 1e-3:  # noqa: PLR2004
        new_beta = 0.5 * (max_beta + min_beta)
        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta - old_beta
        Wm = np.exp(inc_beta * (likelihoods - likelihoods.max()))  # noqa: N806
        ESS = int(1 / np.sum((Wm / sum(Wm)) ** 2))  # noqa: N806
        if rN == ESS:
            break
        elif rN > ESS:  # noqa: RET508
            max_beta = new_beta
        else:
            min_beta = new_beta

    if new_beta < 1e-3:  # noqa: PLR2004
        new_beta = 1e-3
        inc_beta = new_beta - old_beta
        Wm = np.exp(inc_beta * (likelihoods - likelihoods.max()))  # noqa: N806

    if new_beta >= 0.95:  # noqa: PLR2004
        new_beta = 1
        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta - old_beta
        Wm = np.exp(inc_beta * (likelihoods - likelihoods.max()))  # noqa: N806

    return new_beta, Wm, ESS


def compute_beta_evidence_old(  # noqa: D103
    beta,
    log_likelihoods,
    log_evidence,
    prev_ESS,  # noqa: N803
    threshold,
):
    old_beta = beta
    min_beta = beta
    max_beta = 2.0

    N = len(log_likelihoods)  # noqa: N806
    min_ESS = np.ceil(0.1 * N)  # noqa: N806
    rN = max(threshold * prev_ESS, min_ESS)  # noqa: N806

    new_beta = 0.5 * (max_beta + min_beta)
    inc_beta = new_beta - old_beta
    log_Wm = inc_beta * log_likelihoods  # noqa: N806
    log_Wm_n = log_Wm - logsumexp(log_Wm)  # noqa: N806
    ESS = int(np.exp(-logsumexp(log_Wm_n * 2)))  # noqa: N806

    while max_beta - min_beta > 1e-6:  # min step size  # noqa: PLR2004
        new_beta = 0.5 * (max_beta + min_beta)
        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta - old_beta

        log_Wm = inc_beta * log_likelihoods  # noqa: N806
        log_Wm_n = log_Wm - logsumexp(log_Wm)  # noqa: N806
        ESS = int(np.exp(-logsumexp(log_Wm_n * 2)))  # noqa: N806

        if rN == ESS:
            break
        elif rN > ESS:  # noqa: RET508
            max_beta = new_beta
        else:
            min_beta = new_beta

    if new_beta >= 1:
        new_beta = 1
        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta - old_beta

        log_Wm = inc_beta * log_likelihoods  # noqa: N806
        log_Wm_n = log_Wm - logsumexp(log_Wm)  # noqa: N806

    Wm = np.exp(log_Wm)  # noqa: N806, F841
    Wm_n = np.exp(log_Wm_n)  # noqa: N806

    # update model evidence
    # evidence = evidence * (sum(Wm)/N)
    log_evidence = log_evidence + logsumexp(log_Wm) - np.log(N)
    # log_evidence = log_evidence + np.log((sum(Wm)/N))

    return new_beta, log_evidence, Wm_n, ESS


# MCMC
def MCMC_MH_old(  # noqa: N802, D103, PLR0913
    ParticleNum,  # noqa: N803
    Em,  # noqa: N803
    Nm_steps,  # noqa: N803
    current,
    likelihood_current,
    posterior_current,
    beta,
    numAccepts,  # noqa: N803
    AllPars,  # noqa: N803
    log_likelihood,
    variables,
    resultsLocation,  # noqa: N803
    rng,
    calibrationData,  # noqa: N803
    numExperiments,  # noqa: N803
    covarianceMatrixList,  # noqa: N803
    edpNamesList,  # noqa: N803
    edpLengthsList,  # noqa: N803
    normalizingFactors,  # noqa: N803
    locShiftList,  # noqa: N803
    workflowDriver,  # noqa: N803
    prediction_current,
):
    all_proposals = []
    all_PLP = []  # noqa: N806

    # deltas = propose(np.zeros(len(current)), Em, Nm_steps)
    deltas = rng.multivariate_normal(np.zeros(len(current)), Em, Nm_steps)

    for j2 in range(Nm_steps):
        delta = deltas[j2]
        proposal = current + delta
        prior_proposal = log_prior(proposal, AllPars)

        if np.isfinite(prior_proposal):  # proposal satisfies the prior constraints
            # likelihood_proposal = log_likelihood(ParticleNum, proposal, variables, resultsLocation)
            likelihood_proposal, prediction_proposal = runFEM(
                ParticleNum,
                proposal,
                variables,
                resultsLocation,
                log_likelihood,
                calibrationData,
                numExperiments,
                covarianceMatrixList,
                edpNamesList,
                edpLengthsList,
                normalizingFactors,
                locShiftList,
                workflowDriver,
            )

            if np.isnan(likelihood_proposal):
                likelihood_proposal = -np.inf
            posterior_proposal = prior_proposal + likelihood_proposal * beta
        else:
            likelihood_proposal = -np.inf  # dont run the FE model
            posterior_proposal = -np.inf
            prediction_proposal = -np.inf * np.ones_like(prediction_current)

        log_acceptance = posterior_proposal - posterior_current
        all_proposals.append(proposal)
        all_PLP.append([prior_proposal, likelihood_proposal, posterior_proposal])

        # if np.isfinite(log_acceptance) and (np.log(np.random.uniform()) < log_acceptance):
        if np.isfinite(log_acceptance) and (np.log(rng.uniform()) < log_acceptance):
            # accept
            current = proposal
            posterior_current = posterior_proposal
            likelihood_current = likelihood_proposal
            numAccepts += 1  # noqa: N806
            prediction_current = prediction_proposal

    # gather all last samples
    return (
        current,
        likelihood_current,
        posterior_current,
        numAccepts,
        all_proposals,
        all_PLP,
        prediction_current,
    )


# MCMC
def MCMC_MH(  # noqa: N802, D103, PLR0913
    ParticleNum,  # noqa: N803
    Em,  # noqa: N803
    Nm_steps,  # noqa: N803
    current,
    likelihood_current,
    posterior_current,
    beta,
    numAccepts,  # noqa: N803
    AllPars,  # noqa: N803
    log_likelihood,
    variables,
    resultsLocation,  # noqa: N803
    rng,
    calibrationData,  # noqa: N803
    numExperiments,  # noqa: N803
    covarianceMatrixList,  # noqa: N803
    edpNamesList,  # noqa: N803
    edpLengthsList,  # noqa: N803
    normalizingFactors,  # noqa: N803
    locShiftList,  # noqa: N803
    workflowDriver,  # noqa: N803
    prediction_current,
):
    all_proposals = []
    all_PLP = []  # noqa: N806

    # deltas = propose(np.zeros(len(current)), Em, Nm_steps)
    deltas = rng.multivariate_normal(np.zeros(len(current)), Em, Nm_steps)

    for j2 in range(Nm_steps):
        delta = deltas[j2]
        proposal = current + delta
        prior_proposal = log_prior(proposal, AllPars)

        if np.isfinite(prior_proposal):  # proposal satisfies the prior constraints
            # likelihood_proposal = log_likelihood(ParticleNum, proposal, variables, resultsLocation)
            likelihood_proposal, prediction_proposal = runFEM(
                ParticleNum,
                proposal,
                variables,
                resultsLocation,
                log_likelihood,
                calibrationData,
                numExperiments,
                covarianceMatrixList,
                edpNamesList,
                edpLengthsList,
                normalizingFactors,
                locShiftList,
                workflowDriver,
            )

            if np.isnan(likelihood_proposal):
                likelihood_proposal = -np.inf
            posterior_proposal = prior_proposal + likelihood_proposal * beta
        else:
            likelihood_proposal = -np.inf  # dont run the FE model
            posterior_proposal = -np.inf
            prediction_proposal = -np.inf * np.ones_like(prediction_current)

        log_acceptance = posterior_proposal - posterior_current
        all_proposals.append(proposal)
        all_PLP.append([prior_proposal, likelihood_proposal, posterior_proposal])

        # if np.isfinite(log_acceptance) and (np.log(np.random.uniform()) < log_acceptance):
        if np.isfinite(log_acceptance) and (np.log(rng.uniform()) < log_acceptance):
            # accept
            current = proposal
            posterior_current = posterior_proposal
            likelihood_current = likelihood_proposal
            numAccepts += 1  # noqa: N806
            prediction_current = prediction_proposal

    # gather all last samples
    return (
        current,
        likelihood_current,
        posterior_current,
        numAccepts,
        all_proposals,
        all_PLP,
        prediction_current,
    )


# def compute_beta_evidence(beta, log_likelihoods, prev_ESS, threshold=0.95):
#     old_beta = beta
#     min_beta = beta
#     max_beta = 2.0
#     N = len(log_likelihoods)
#     rN = max(threshold*prev_ESS, 50)
#     while max_beta - min_beta > 1e-3: #min step size
#         new_beta = 0.5*(max_beta+min_beta)
#         #plausible weights of Sm corresponding to new beta
#         inc_beta = new_beta-old_beta

#         log_Wm = inc_beta * log_likelihoods
#         log_Wm_n = log_Wm - logsumexp(log_Wm)
#         ESS = int(np.exp(-logsumexp(log_Wm_n * 2)))

#         if ESS == rN:
#             break
#         elif ESS < rN:
#             max_beta = new_beta
#         else:
#             min_beta = new_beta

#     if new_beta >= 1:
#         new_beta = 1
#         #plausible weights of Sm corresponding to new beta
#         inc_beta = new_beta-old_beta

#         log_Wm = inc_beta * log_likelihoods
#         log_Wm_n = log_Wm - logsumexp(log_Wm)

#     Wm = np.exp(log_Wm)
#     Wm_n = np.exp(log_Wm_n)

#     # update model evidence
#     # evidence = evidence * (sum(Wm)/N)
#     log_evidence = logsumexp(log_Wm) - np.log(N)
#     # log_evidence = log_evidence + np.log((sum(Wm)/N))

#     return new_beta, log_evidence, Wm_n, ESS


def get_weights(dBeta, log_likelihoods):  # noqa: N803, D103
    log_weights = dBeta * log_likelihoods
    log_sum_weights = logsumexp(log_weights)
    log_weights_normalized = log_weights - log_sum_weights
    weights_normalized = np.exp(log_weights_normalized)
    std_weights_normalized = np.std(weights_normalized)
    cov_weights = np.std(weights_normalized) / np.mean(weights_normalized)

    return weights_normalized, cov_weights, std_weights_normalized


def compute_beta_evidence(beta, log_likelihoods, logFile, threshold=1.0):  # noqa: N803, D103
    max_beta = 1.0
    dBeta = min(max_beta, 1.0 - beta)  # noqa: N806

    weights, cov_weights, std_weights = get_weights(dBeta, log_likelihoods)

    while cov_weights > (threshold) or (std_weights == 0):
        dBeta = dBeta * 0.99  # noqa: N806

        # while (cov_weights > (threshold+0.00000005) or (std_weights == 0)):
        #     if ((cov_weights > (threshold+1.0)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.9
        #     if ((cov_weights > (threshold+0.5)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.95
        #     if ((cov_weights > (threshold+0.05)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.99
        #     if ((cov_weights > (threshold+0.005)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.999
        #     if ((cov_weights > (threshold+0.0005)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.9999
        #     if ((cov_weights > (threshold+0.00005)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.99999
        #     if ((cov_weights > (threshold+0.000005)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.999999
        #     if ((cov_weights > (threshold+0.0000005)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.9999999
        #     if ((cov_weights > (threshold+0.00000005)) or  (std_weights == 0)):
        #         dBeta = dBeta*0.99999999

        # if dBeta < 1e-3:
        #     dBeta = 1e-3
        #     weights, cov_weights, std_weights = get_weights(dBeta, log_likelihoods)
        #     break
        weights, cov_weights, std_weights = get_weights(dBeta, log_likelihoods)

    beta = beta + dBeta
    if beta > 0.95:  # noqa: PLR2004
        beta = 1
    log_evidence = logsumexp(dBeta * log_likelihoods) - np.log(len(log_likelihoods))

    try:
        ESS = int(1 / np.sum((weights / np.sum(weights)) ** 2))  # noqa: N806
    except OverflowError as err:
        ESS = 0  # noqa: N806
        logFile.write(str(err))

    return beta, log_evidence, weights, ESS
