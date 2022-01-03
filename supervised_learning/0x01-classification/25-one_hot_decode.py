#!/usr/bin/env python3
"""24. One-Hot Encode"""

import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
    classes is the maximum number of classes
    - m is the number of examples
    Returns: a numpy.ndarray with shape (m, ) containing the numeric labels
        for each example, or None on failure
    """
    if type(one_hot) is not np.ndarray:
        return None
    try:
        one_hot = np.argmax(one_hot, axis=0)
        return one_hot
    except Exception as error:
        return None
