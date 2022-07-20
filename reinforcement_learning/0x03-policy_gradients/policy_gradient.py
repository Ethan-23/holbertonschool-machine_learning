#!/usr/bin/env python3
"""Simple Policy function"""

import numpy as np


def policy(matrix, weight):
    """computes to policy with a weight of a matrix"""
    dot = matrix.dot(weight)
    exp = np.exp(dot)
    res = exp / np.sum(exp)
    return(res)


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy gradient
       based on a state and a weight matrix"""
    p = policy(state, weight)
    action = np.random.choice(len(p[0]), p=p[0])
    s = p.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    dsoftmax = softmax[action, :]
    dlog = dsoftmax / p[0, action]
    gradient = state.T.dot(dlog[None, :])
    return action, gradient
