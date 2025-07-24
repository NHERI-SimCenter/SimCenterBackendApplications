"""
Provides a class for performing Latin Hypercube Sampling (LHS).

This module generates space-filling designs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, PositiveInt
from scipy.stats import norm, qmc


class LatinHypercubeSampling(BaseModel, validate_assignment=True):
    """
    A class to perform Latin Hypercube Sampling (LHS) for generating space-filling designs.

    Attributes
    ----------
        n_samples (PositiveInt): The number of samples to generate.
        n_dimensions (PositiveInt): The number of dimensions for each sample.
        seed (Optional[PositiveInt]): The seed for random number generation.
    """

    n_samples: PositiveInt
    n_dimensions: PositiveInt
    seed: Optional[PositiveInt] = None  # noqa: UP007

    def generate(self):
        """
        Generate samples using Latin Hypercube Sampling.

        Returns
        -------
            np.ndarray: The generated samples.
        """
        lhs = qmc.LatinHypercube(
            self.n_dimensions, seed=self.seed, optimization='random-cd'
        )
        samples = lhs.random(self.n_samples)
        samples = norm.ppf(samples)  # Transform to standard normal distribution
        return samples  # noqa: RET504


if __name__ == '__main__':
    """
    Example usage of the LatinHypercubeSampling class.
    """
    # initial_doe = LatinHypercubeSampling(n_samples=10, n_dimensions=2)
    # print(repr(initial_doe))
    # samples = initial_doe.generate()
    # print(samples)
    # domain = [(0, 10), (-5, 5)]  # Example of domain for each dimension
    # scaled_samples = initial_doe.generate(domain)
    # print(scaled_samples)
    # # initial_doe.seed = 0
    # # idoe = InitialDesignOfExperiments(n_samples=0, n_dimensions=0, seed=-1)
