"""Created on Mon Oct 24 18:10:31 2022

@author: snaeimi
"""  # noqa: INP001, D400, D415

# import numba  # noqa: ERA001
import operator
from functools import reduce  # Valid in Python 2.6+, required in Python 3

import numpy as np


def hhelper(x):  # noqa: ANN001, ANN201, D103
    if x < 0:
        return 0
    else:  # noqa: RET505
        return x


# @numba.jit()
def EPHelper(prob_mat, old):  # noqa: ANN001, ANN201, N802, D103
    if old == False:  # prob_mat = prob_mat.tolist()  # noqa: E712
        # one_minus_p_list = 1-prob_mat  # noqa: ERA001
        one_minus_p_list = [1 - p for p in prob_mat]
        pi_one_minus_p_list = [
            1 - reduce(operator.mul, one_minus_p_list[: i + 1], 1)
            for i in range(len(one_minus_p_list))
        ]
        # pi_one_minus_p_list         = [rr.apply(lambda x: [x[i] * x[1], raw=True)
        return pi_one_minus_p_list  # noqa: RET504
        # pi_one_minus_p_list.iloc[0] =  one_minus_p_list.iloc[0]  # noqa: ERA001

        # return (pd.Series(1.00, index=pi_one_minus_p_list.index) - pi_one_minus_p_list, prob_mat)  # noqa: ERA001, E501
    else:  # noqa: RET505
        ep_mat = np.ndarray(prob_mat.size)
        for i in np.arange(prob_mat.size):
            j = 0
            pi_one_minus_p = 1
            while j <= i:
                p = prob_mat[j]
                one_minus_p = 1 - p
                pi_one_minus_p *= one_minus_p
                j += 1
            ep_mat[i] = 1 - pi_one_minus_p
    return ep_mat


def helper_outageMap(pandas_list):  # noqa: ANN001, ANN201, N802, D103
    false_found_flag = False
    b_list = pandas_list.tolist()
    i = 0
    for b_value in b_list:
        if b_value == False:  # noqa: E712
            false_found_flag = True
            break
        i += 1

    return false_found_flag, i


def hhelper(x):  # noqa: ANN001, ANN201, D103, F811
    if x < 0:
        return 0
    else:  # noqa: RET505
        return x
