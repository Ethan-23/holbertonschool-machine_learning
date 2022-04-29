#!/usr/bin/env python3
"""0x0D. RNNs"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """RNN forward prop"""
    H = np.ndarray((X.shape[0] + 1, X.shape[1], h_0.shape[1]))
    Y = np.ndarray((X.shape[0], X.shape[1], rnn_cell.Wy.shape[1]))
    H[0] = h_0
    for time in range(X.shape[0]):
        H[time + 1], Y[time] = rnn_cell.forward(H[time], X[time])
    return H, Y
