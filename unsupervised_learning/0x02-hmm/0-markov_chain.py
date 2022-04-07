#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    markov_chain
    """
    for i in range(t):
        s = np.matmul(s, P)
    return s
