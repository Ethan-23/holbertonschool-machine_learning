#!/usr/bin/env python3
"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X is a numpy.ndarray of shape (nx, m) containing the input data for
        the network
    - nx is the number of input features
    - m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    Returns: a dictionary containing the outputs of each layer and the
        dropout mask used on each layer (see example for format)
    """
    output = {}
    output["A0"] = X
    for i in range(L):
        weight = weights["W{}".format(i + 1)]
        bias = weights["b{}".format(i + 1)]
        z = np.matmul(weight, output["A{}".format(i)]) + bias
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        if i != (L - 1):
            A = (np.tanh(z) * dropout) / keep_prob
            output["D{}".format(i + 1)] = dropout
        else:
            A = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
        output["A{}".format(i + 1)] = A
    return output
