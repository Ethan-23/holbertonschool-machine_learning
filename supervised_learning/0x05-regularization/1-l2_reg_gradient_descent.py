#!/usr/bin/env python3
"""L2 Regularization Cost"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
    - classes is the number of classes
    - m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    """
    weights_copy = weights.copy()
    m = Y.shape[1]
    dz = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        w = "W{}".format(i)
        b = "b{}".format(i)
        A = cache["A{}".format(i - 1)]
        dw = (1/m) * np.matmul(dz, A.transpose()) + (lambtha * weights[w]) / m
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        weights[w] = weights[w] - (alpha * dw)
        weights[b] = weights[b] - (alpha * db)
        dz = np.matmul(weights_copy[w].transpose(), dz) * (1 - A * A)
