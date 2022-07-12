#!/usr/bin/env python3
"""Simple Policy function"""

import numpy as np

def policy(matrix, weight):
    """computes to policy with a weight of a matrix"""
    dot = matrix.dot(weight)
    exp = np.exp(dot)
    res = exp / np.sum(exp)
    return(res)
