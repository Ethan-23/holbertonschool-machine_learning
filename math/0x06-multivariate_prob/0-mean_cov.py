#!/usr/bin/env python3
"""0. Mean and Covariance"""

import numpy as np


def mean_cov(X):
    """Returns: mean, cov"""
    if type(X) is not np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    cov = (1 / (n - 1)) * np.matmul(X.T - mean.T, X - mean)
    return mean, cov
