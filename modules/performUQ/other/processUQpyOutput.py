from pathlib import Path

import numpy as np


def output_function(index):
    filePath = Path('./results.out').resolve()
    return np.atleast_2d(np.genfromtxt(filePath))
