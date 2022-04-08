#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """The Forward Algorithm"""
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != Transition.shape[1] or Transition.shape[0] != N:
        return (None, None)
    if N != Initial.shape[0] or Initial.shape[1] != 1:
        return (None, None)
    F = np.zeros([N, T])
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            F[j, i] = F[:, i - 1].dot(Transition[:, j]) *\
                Emission[j, Observation[i]]
    P = np.sum(F[:, T - 1:], axis=0)[0]
    return (P, F)
