#!/usr/bin/env python3
"""0x0D. RNNs"""

import numpy as np


class GRUCell:
    """GRUCell Class"""
    def __init__(self, i, h, o):
        """init for GRUCell class"""
        self.Wz = np.random.normal(size=(h+i, h))
        self.Wr = np.random.normal(size=(h+i, h))
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward for GRUCell"""
        concat = np.concatenate((h_prev, x_t), 1)
        exp = np.exp(np.matmul(concat, self.Wr) + self.br)
        sig1 = exp / (1 + exp)
        exp = np.exp(np.matmul(concat, self.Wz) + self.bz)
        sig2 = exp / (1 + exp)
        concat = np.concatenate((sig1 * h_prev, x_t), 1)
        tanh = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        h_next = (1 - sig2) * h_prev + sig2 * tanh
        y = np.exp(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y / y.sum(axis=1, keepdims=True)
