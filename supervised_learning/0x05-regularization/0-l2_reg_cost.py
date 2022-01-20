#!/usr/bin/env python3
"""L2 Regularization Cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost is the cost of the network without L2 regularization
    lambtha is the regularization parameter
    weights is a dictionary of the weights and biases (numpy.ndarrays)
        of the neural network
    L is the number of layers in the neural network
    m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    weights_together = 0
    for i in range(1, L + 1):
        w = weights["W{}".format(i)]
        weights_together += np.linalg.norm(w)
    l2Cost = cost + lambtha * weights_together / (2 * m)
    return l2Cost
