#!/usr/bin/env python3
"""0-norm_constraints"""
import numpy as np


def normalization_constants(X):
    """
    X is the numpy.ndarray of shape (m, nx) to normalize
    - m is the number of data points
    - nx is the number of features
    Returns: the mean and standard deviation of each feature, respectively
    """
    sdev = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    return mean, sdev
