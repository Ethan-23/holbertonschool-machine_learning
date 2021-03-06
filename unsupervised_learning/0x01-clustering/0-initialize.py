#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
        that will be used for K-means clustering
    - n is the number of data points
    - d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    """
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    high = np.max(X, axis=0)
    low = np.min(X, axis=0)
    return np.random.uniform(low=low, high=high, size=(k, d))
