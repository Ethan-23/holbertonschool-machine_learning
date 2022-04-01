#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Optimize k
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) is not int or type(kmax) is not int:
        return None, None
    if kmin < 1 or kmax < 1:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    result = []
    vars = []
    for i in range(kmin, kmax + 1):
        C, clss = kmeans(X, i, iterations)
        result.append((C, clss))
        var = variance(X, result[0][0]) - variance(X, C)
        vars.append(var)
    return result, vars
