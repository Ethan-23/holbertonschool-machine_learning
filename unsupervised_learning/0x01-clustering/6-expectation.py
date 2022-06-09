#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    expectation
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if not np.isclose(pi.sum(), 1):
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if S.shape[2] != S.shape[1]:
        return (None, None)
    if S.shape[0] != pi.shape[0] or S.shape[0] != m.shape[0]:
        return (None, None)
    if m.shape[1] != X.shape[1]:
        return (None, None)
    n, d = X.shape
    k = pi.shape[0]
    results = np.zeros([k, n])
    for i in range(k):
        P = pdf(X, m[i], S[i])
        results[i] = pi[i] * P
    likelihood = np.sum(np.log(results.sum(axis=0)))
    results = results / results.sum(axis=0)
    return (results, likelihood)
