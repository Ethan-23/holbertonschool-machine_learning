#!/usr/bin/env python3
"""0x0D. RNNs"""

import numpy as np


class BidirectionalCell:
    """BidirectionalCell Class"""
    def __init__(self, i, h, o):
        """init for BidirectionalCell class"""
        self.Whf = np.random.normal(size=(h+i, h))
        self.Whb = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h * 2, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward for BidirectionalCell"""
        concat = np.concatenate((h_prev, x_t), 1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """backward for BidirectionalCell"""
        concat = np.concatenate((h_next, x_t), 1)
        h_pev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_pev
