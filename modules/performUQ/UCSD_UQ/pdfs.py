"""@author: Mukesh, Maitreya, Conte, Aakash"""

import numpy as np
from scipy import stats


class Dist:
    def __init__(self, dist_name, params=None, moments=None, data=None):
        self.dist_name = dist_name
        self.params = params
        self.moments = moments
        self.data = data
        if (params is None) and (moments is None) and (data is None):
            raise RuntimeError(
                'Atleast one of parameters, moments, or data must be specified when creating a random variable'
            )


class Uniform:
    # Method with in this uniform class
    def __init__(
        self,
        lower,
        upper,
    ):  # method receives instance as first argument automatically
        # the below are the instance variables
        self.lower = lower
        self.upper = upper

    # Method to generate random numbers
    def generate_rns(self, N):
        return (self.upper - self.lower) * np.random.rand(N) + self.lower

    # Method to compute log of the pdf at x
    def log_pdf_eval(self, x):
        if (x - self.upper) * (x - self.lower) <= 0:
            lp = np.log(1 / (self.upper - self.lower))
        else:
            lp = -np.inf
        return lp


class Halfnormal:
    def __init__(self, sig):
        self.sig = sig

    def generate_rns(self, N):
        return self.sig * np.abs(np.random.randn(N))

    def log_pdf_eval(self, x):
        if x >= 0:
            lp = (
                -np.log(self.sig)
                + 0.5 * np.log(2 / np.pi)
                - ((x * x) / (2 * self.sig * self.sig))
            )
        else:
            lp = -np.inf
        return lp


class Normal:
    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig

    def generate_rns(self, N):
        return self.sig * np.random.randn(N) + self.mu

    def log_pdf_eval(self, x):
        lp = (
            -0.5 * np.log(2 * np.pi)
            - np.log(self.sig)
            - 0.5 * (((x - self.mu) / self.sig) ** 2)
        )
        return lp


class TrunNormal:
    def __init__(self, mu, sig, a, b):
        self.mu = mu
        self.sig = sig
        self.a = a
        self.b = b

    def generate_rns(self, N):
        return stats.truncnorm(
            (self.a - self.mu) / self.sig,
            (self.b - self.mu) / self.sig,
            loc=self.mu,
            scale=self.sig,
        ).rvs(N)

    def log_pdf_eval(self, x):
        lp = stats.truncnorm(
            (self.a - self.mu) / self.sig,
            (self.b - self.mu) / self.sig,
            loc=self.mu,
            scale=self.sig,
        ).logpdf(x)
        return lp


class mvNormal:
    def __init__(self, mu, E):
        self.mu = mu
        self.E = E
        self.d = len(mu)
        self.logdetE = np.log(np.linalg.det(self.E))
        self.Einv = np.linalg.inv(E)

    def generate_rns(self, N):
        return np.random.multivariate_normal(self.mu, self.E, N)

    def log_pdf_eval(self, x):
        xc = x - self.mu
        return (
            -(0.5 * self.d * np.log(2 * np.pi))
            - (0.5 * self.logdetE)
            - (0.5 * np.transpose(xc) @ self.Einv @ xc)
        )


class InvGamma:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.dist = stats.invgamma(self.a, scale=self.b)

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class BetaDist:
    def __init__(self, alpha, beta, lowerbound, upperbound):
        self.alpha = alpha
        self.beta = beta
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.dist = stats.beta(
            self.alpha, self.beta, self.lowerbound, self.upperbound
        )

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class LogNormDist:
    def __init__(self, mu, sigma):
        # self.sigma = np.sqrt(np.log(zeta**2/lamda**2 + 1))
        # self.mu = np.log(lamda) - 1/2*self.sigma**2
        self.s = sigma
        self.loc = 0
        self.scale = np.exp(mu)
        self.dist = stats.lognorm(s=self.s, loc=self.loc, scale=self.scale)

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class GumbelDist:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.dist = stats.gumbel_r(loc=self.beta, scale=(1 / self.alpha))

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class WeibullDist:
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        self.dist = stats.weibull_min(c=self.shape, scale=self.scale)

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class ExponentialDist:
    def __init__(self, lamda):
        self.lamda = lamda
        self.scale = 1 / self.lamda
        self.dist = stats.expon(scale=self.scale)

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class TruncatedExponentialDist:
    def __init__(self, lamda, lower, upper):
        self.lower = lower
        self.upper = upper
        self.lamda = lamda
        self.scale = 1 / self.lamda
        self.loc = self.lower
        self.b = (self.upper - self.lower) / self.scale
        self.dist = stats.truncexpon(b=self.b, loc=self.loc, scale=self.scale)

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class GammaDist:
    def __init__(self, k, lamda):
        self.k = k
        self.lamda = lamda
        self.alpha = k
        self.beta = lamda
        self.scale = 1 / self.beta
        self.dist = stats.gamma(a=self.alpha, scale=self.scale)

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class ChiSquareDist:
    def __init__(self, k):
        self.k = k
        self.dist = stats.chi2(k=self.k)

    def generate_rns(self, N):
        return self.dist.rvs(size=N)

    def log_pdf_eval(self, x):
        return self.dist.logpdf(x)


class DiscreteDist:
    def __init__(self, values, weights):
        self.values = values
        self.weights = weights
        self.probabilities = self.weights / np.sum(self.weights)
        self.log_probabilities = np.log(self.weights) - np.log(np.sum(self.weights))
        self.rng = np.random.default_rng()

    def generate_rns(self, N):
        return self.rng.choice(self.values, N, p=self.probabilities)

    def U2X(self, u):
        cumsum_prob = np.cumsum(self.probabilities)
        cumsum_prob = np.insert(cumsum_prob, 0, 0)
        cumsum_prob = cumsum_prob[:-1]
        x = np.zeros_like(u)
        for i, u_comp in enumerate(u):
            cdf_val = stats.norm.cdf(u_comp)
            x[i] = self.values[np.where(cumsum_prob <= cdf_val)[0][-1]]
        return x

    def log_pdf_eval(self, u):
        x = self.U2X(u)
        lp = np.zeros_like(x)
        for i, x_comp in enumerate(x):
            lp[i] = self.log_probabilities[np.where(self.values == x_comp)]
        return lp


class ConstantInteger:
    def __init__(self, value) -> None:
        self.value = value

    def generate_rns(self, N):
        return np.array([self.value for _ in range(N)], dtype=int)

    def log_pdf_eval(self, x):
        return 0.0
