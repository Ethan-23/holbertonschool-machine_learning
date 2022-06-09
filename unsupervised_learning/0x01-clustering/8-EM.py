#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """expectation_maximization"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None, None)
    if type(k) is not int or type(iterations) is not int:
        return (None, None, None, None, None)
    if k <= 0 or iterations <= 0:
        return (None, None, None, None, None)
    if type(tol) is not float or tol < 0:
        return (None, None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None, None)
    n, d = X.shape
    pi, m, S = initialize(X, k)
    results, lh = expectation(X, pi, m, S)
    lh_prev = 0
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format
                  (i, lh.round(5)))
        pi, m, S = maximization(X, results)
        results, lh = expectation(X, pi, m, S)
        if np.abs(lh_prev - lh) <= tol:
            break
        lh_prev = lh
    if verbose:
        print("Log Likelihood after {} iterations: {}".format
              (i + 1, lh.round(5)))
    return(pi, m, S, results, lh)
