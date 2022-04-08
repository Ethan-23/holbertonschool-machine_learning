#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def absorbing(P):
    """
    markov_chain
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if np.min(P ** 2) < 0 or np.min(P ** 3) < 0:
        return False
    ab_state = np.where(np.diag(P) == 1)
    if len(ab_state[0]) == P.shape[0]:
        return True
    if len(ab_state[0]) == 0:
        return False
    return True
