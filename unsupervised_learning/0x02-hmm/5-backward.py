#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """The Backward Algorithm"""
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != Transition.shape[1] or Transition.shape[0] != N:
        return (None, None)
    if N != Initial.shape[0] or Initial.shape[1] != 1:
        return (None, None)
    B = np.zeros([N, T])
    B[:, T - 1] = np.ones((N))
    for i in range(T - 2, -1, -1):
        for j in range(N):
            B[j, i] = (B[:, i + 1] *
                       Emission[:, Observation[i + 1]]).dot(Transition[j, :])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return (P, B)
