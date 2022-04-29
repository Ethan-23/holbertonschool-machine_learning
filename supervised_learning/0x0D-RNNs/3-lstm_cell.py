#!/usr/bin/env python3
"""0x0D. RNNs"""

import numpy as np


def sigmoid(num):
    """sigmoid function"""
    ex = np.exp(num)
    return ex / (1 + ex)


class LSTMCell:
    """LSTMCell Class"""
    def __init__(self, i, h, o):
        """init for LSTMCell class"""
        self.Wf = np.random.normal(size=(h+i, h))
        self.Wu = np.random.normal(size=(h+i, h))
        self.Wc = np.random.normal(size=(h+i, h))
        self.Wo = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """forward for LSTMCell"""
        concat = np.concatenate((h_prev, x_t), 1)
        sig1 = sigmoid(np.matmul(concat, self.Wf) + self.bf)
        sig2 = sigmoid(np.matmul(concat, self.Wu) + self.bu)
        tanh = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        sig3 = sigmoid(np.matmul(concat, self.Wo) + self.bo)
        c_next = sig1 * c_prev + sig2 * tanh
        h_next = sig3 * np.tanh(c_next)
        y = np.exp(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y / y.sum(axis=1, keepdims=True)
