#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """The Viretbi Algorithm"""
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != Transition.shape[1] or Transition.shape[0] != N:
        return (None, None)
    if N != Initial.shape[0] or Initial.shape[1] != 1:
        return (None, None)
    D = np.zeros([N, T])
    path = np.zeros(T)
    phi = np.zeros([N, T])
    D[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            D[j, i] = np.max(D[:, i - 1] * Transition[:, j]) *\
                Emission[j, Observation[i]]
            phi[j, i] = np.argmax(D[:, i-1] * Transition[:, j])
    path[T - 1] = np.argmax(D[:, T - 1])
    for i in range(T-2, -1, -1):
        path[i] = phi[int(path[i + 1]), i + 1]
    P = np.max(D[:, T - 1:], axis=0)[0]
    path = [int(i) for i in path]
    return (path, P)