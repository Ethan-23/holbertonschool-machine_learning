#!/usr/bin/env python3
"""24. One-Hot Encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Y is a numpy.ndarray with shape (m,) containing numeric class labels
    - m is the number of examples
    classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m), or None
        on failure
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        size = (classes, len(Y))
        one_hot = np.zeros(size)
        rows = np.arange(classes)
        one_hot[Y, rows] = 1
        return one_hot
    except Exception as error:
        return None
