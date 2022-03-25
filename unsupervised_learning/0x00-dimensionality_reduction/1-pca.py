#!/usr/bin/env python3
"""PCA"""

import numpy as np


def pca(X, ndim):
    """
    X is a numpy.ndarray of shape (n, d) where:
    - n is the number of data points
    - d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim)
        containing the transformed version of X
    """
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    i, j, k = np.linalg.svd(A)
    W = k.T[:, :ndim]
    T = np.matmul(A, W)
    return (T)
