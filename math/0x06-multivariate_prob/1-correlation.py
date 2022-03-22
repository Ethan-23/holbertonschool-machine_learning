#!/usr/bin/env python3
"""1. Correlation"""

import numpy as np


def correlation(C):
    """
    Returns a numpy.ndarray of shape (d, d)
        containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    d, d2 = C.shape
    if d != d2:
        raise ValueError("C must be a 2D square matrix")
    corr = (1 / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))) * C
    return (corr)
