# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:10:31 2022

@author: snaeimi
"""

#import numba
import pandas as pd
import numpy as np
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

def hhelper(x):
    if x<0:
        return 0
    else:
        return x
    
#@numba.jit()
def EPHelper(prob_mat, old):
    if old==False:#prob_mat = prob_mat.tolist()
        #one_minus_p_list = 1-prob_mat
        one_minus_p_list = [1-p for p in prob_mat]
        pi_one_minus_p_list = [1- reduce(operator.mul, one_minus_p_list[:i+1], 1) for i in range(0, len(one_minus_p_list))]
        #pi_one_minus_p_list         = [rr.apply(lambda x: [x[i] * x[1], raw=True) 
        return pi_one_minus_p_list
        #pi_one_minus_p_list.iloc[0] =  one_minus_p_list.iloc[0]
        
        #return (pd.Series(1.00, index=pi_one_minus_p_list.index) - pi_one_minus_p_list, prob_mat)
    else:
        ep_mat = np.ndarray(prob_mat.size)
        for i in np.arange(prob_mat.size):
            j=0
            pi_one_minus_p = 1
            while j <= i:
                p = prob_mat[j]
                one_minus_p = 1 - p
                pi_one_minus_p *= one_minus_p
                j += 1
            ep_mat[i] = 1- pi_one_minus_p
    return ep_mat

def helper_outageMap(pandas_list):
    false_found_flag = False
    b_list = pandas_list.tolist()
    i = 0
    for b_value in b_list:
       if b_value == False:
           false_found_flag = True
           break
       i += 1
   
    return  false_found_flag, i

def hhelper(x):
    if x<0:
        return 0
    else:
        return x