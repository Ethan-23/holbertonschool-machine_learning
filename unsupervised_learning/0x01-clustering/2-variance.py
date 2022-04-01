#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def variance(X, C):
    """
    variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[0] < C.shape[0] or X.shape[1] != C.shape[1]:
        return None
    distance = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    clss = np.min(distance, axis=0)
    return (clss**2).sum().sum()
