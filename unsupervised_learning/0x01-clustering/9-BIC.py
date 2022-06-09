#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """BIC Function"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None, None, None)
    if type(kmin) is not int or type(kmax) is not int:
        return (None, None, None, None)
    if kmax <= 0 or kmax >= X.shape[0]:
        return (None, None, None, None)
    if kmin <= 0 or kmin >= X.shape[0] or kmin >= kmax:
        return (None, None, None, None)
    if type(tol) is not float or tol <= 0:
        return (None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None)
    n, d = X.shape
    best_k = []
    best_result = []
    l1 = []
    b = []
    for i in range(kmin, kmax + 1):
        pi, m, S, g, lh = expectation_maximization(X, i, iterations,
                                                   tol, verbose)

        temp = (d * i) + (i * d * (d + 1) / 2) + i - 1
        l1.append(lh)
        best_k.append(i)
        best_result.append((pi, m, S))
        BIC = temp * np.log(n) - 2 * lh
        b.append(BIC)
    l1 = np.array(l1)
    b = np.array(b)
    best = np.argmin(b)
    return (best_k[best], best_result[best], l1, b)
