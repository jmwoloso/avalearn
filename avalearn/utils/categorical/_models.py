"""
_models.py : modeling routines for determining variable significance
"""
import numpy as np

def logit(x):
    return np.log(x/(1-x))