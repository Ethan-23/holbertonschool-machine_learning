#!/usr/bin/env python3
"""1. Regular Chains"""

import numpy as np


def regular(P):
    """
    Regular Chains
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, n_check = P.shape
    if n != n_check:
        return None
    if not (P > 0).all():
        return None
    Identity = np.identity(n)
    Q = P - Identity
    one = np.ones((n,))
    Qone = np.c_[Q, one]
    Qmul = np.matmul(Qone, Qone.T)
    Qone2 = np.ones((n,))
    result = np.linalg.solve(Qmul, Qone2)
    return np.expand_dims(result, axis=0)
