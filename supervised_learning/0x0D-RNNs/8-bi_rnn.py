#!/usr/bin/env python3
"""0x0D. RNNs"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """bi_rnn"""
    T = X.shape[0]
    h_forward = []
    h_backward = []
    for i in range(T):
        h_0 = bi_cell.forward(h_0, X[i])
        h_t = bi_cell.backward(h_t, X[T - 1 - i])
        h_forward.append(h_0)
        h_backward.append(h_t)
    h_backward = [x for x in reversed(h_backward)]
    h_forward = np.array(h_forward)
    h_backward = np.array(h_backward)
    H = np.concatenate((h_forward, h_backward), axis=-1)
    Y = bi_cell.output(H)
    return (H, Y)
