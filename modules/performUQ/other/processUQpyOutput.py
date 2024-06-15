import numpy as np
from pathlib import Path

def output_function(index):
    filePath = Path('./results.out').resolve()
    return np.atleast_2d(np.genfromtxt(filePath))