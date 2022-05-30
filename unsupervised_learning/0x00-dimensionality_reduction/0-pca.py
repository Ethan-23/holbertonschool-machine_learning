#!/usr/bin/env python3
"""PCA documentation"""

import numpy as np


def pca(X, var=0.95):
    """
    X is a numpy.ndarray of shape (n, d) where:
        - n is the number of data points
        - d is the number of dimensions in each point
        - all dimensions have a mean of 0 across all
            data points
    var is the fraction of the variance that the PCA
        transformation should maintain
    Returns: the weights matrix, W, that maintains var
        fraction of X‘s original variance
    W is a numpy.ndarray of shape (d, nd) where nd is
        the new dimensionality of the transformed X
    """
    i, j, k = np.linalg.svd(X)
    ratios = list(x / np.sum(j) for x in j)
    variance = np.cumsum(ratios)
    nd = np.argwhere(variance >= var)[0, 0]
    W = k.T[:, :(nd + 1)]
    return (W)
