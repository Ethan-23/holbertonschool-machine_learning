#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Optimize k
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k < 1:
        return None, None, None
    n, d = X.shape
    pi = np.ones((k)) / k
    m, clss = kmeans(X, k)
    S = np.zeros((k, d, d))
    S[:] = np.eye(d, d)
    return pi, m, S
