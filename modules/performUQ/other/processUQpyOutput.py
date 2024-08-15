from pathlib import Path  # noqa: INP001, D100

import numpy as np


def output_function(index):  # noqa: ARG001, D103
    filePath = Path('./results.out').resolve()  # noqa: N806
    return np.atleast_2d(np.genfromtxt(filePath))
