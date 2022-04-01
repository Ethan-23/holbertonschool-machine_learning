#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    - n is the number of data points
    - d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number
        of iterations that should be performed
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    high = np.max(X, axis=0)
    low = np.min(X, axis=0)
    C = np.random.uniform(low=low, high=high, size=(k, d))
    distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    clss = np.argmin(distances, axis=0)
    for i in range(iterations):
        copy = np.copy(C)
        for j in range(k):
            idx = np.where(clss == j)
            if len(idx[0]) == 0:
                C[j] = np.random.uniform(low=low, high=high, size=(1, d))
            else:
                C[j] = X[idx].mean(axis=0)
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        clss = np.argmin(distances, axis=0)
        if np.array_equal(copy, C):
            break
    return C, clss
