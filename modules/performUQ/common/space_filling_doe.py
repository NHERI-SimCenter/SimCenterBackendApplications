from typing import Optional

import numpy as np
from pydantic import BaseModel, PositiveInt
from scipy.stats import qmc


class LatinHypercubeSampling(BaseModel, validate_assignment=True):
    n_samples: PositiveInt
    n_dimensions: PositiveInt
    seed: Optional[PositiveInt] = None

    def generate(self, domain=None):
        lhs = qmc.LatinHypercube(self.n_dimensions, seed=self.seed)
        samples = lhs.random(self.n_samples)
        if domain is not None:
            # Scale samples to the provided domain
            lower_bounds = np.array([bound[0] for bound in domain])
            upper_bounds = np.array([bound[1] for bound in domain])
            samples = qmc.scale(samples, lower_bounds, upper_bounds)
        return samples


if __name__ == "__main__":
    initial_doe = LatinHypercubeSampling(n_samples=10, n_dimensions=2)
    print(repr(initial_doe))
    samples = initial_doe.generate()
    print(samples)
    domain = [(0, 10), (-5, 5)]  # Example of domain for each dimension
    scaled_samples = initial_doe.generate(domain)
    print(scaled_samples)
    # initial_doe.seed = 0
    # idoe = InitialDesignOfExperiments(n_samples=0, n_dimensions=0, seed=-1)
