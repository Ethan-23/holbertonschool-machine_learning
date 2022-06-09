#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def maximization(X, g):
    """
    maximization
    """
    print(g)
    n, d = X.shape
    k = g.shape[0]

    pi = g.sum(axis=1) / n
    m = np.dot(g, X) / g.sum(1)[:, None]
    S = np.zeros([k, d, d])
    for i in range(k):
        temp = X - m[i, :]
        S[i] = (g[i, :, None, None] *
                np.matmul(temp[:, :, None], temp[:, None, :])).sum(axis=0)
    S = S / g.sum(axis=1)[:, None, None]

    return pi, m, S
