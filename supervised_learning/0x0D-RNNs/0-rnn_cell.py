#!/usr/bin/env python3
"""0x0D. RNNs"""

import numpy as np


class RNNCell:
    """RNNCell Class"""
    def __init__(self, i, h, o):
        """init for RNNCell class"""
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward for RNNCell"""
        concat = np.concatenate((h_prev, x_t), 1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y = np.exp(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y / y.sum(axis=1, keepdims=True)
