#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
    - classes is the number of classes
    - m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs and dropout masks of each layer of
        the neural network
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    """
    weights_copy = weights.copy()
    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        if i < L:
            dZ = np.matmul(weights_copy[w].transpose(), dZ) * (1 - A * A)
            dZ *= cache["D{}".format(i)] / keep_prob
        w = "W{}".format(i)
        b = "b{}".format(i)
        A = cache["A{}".format(i - 1)]
        dw = (1/m) * np.matmul(dZ, A.transpose())
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        weights[w] = weights[w] - (alpha * dw)
        weights[b] = weights[b] - (alpha * db)
