#!/usr/bin/env python3
"""Convolutional Back Prop"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the output of the pooling layer
    - m is the number of examples
    - h_new is the height of the output
    - w_new is the width of the output
    - c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
        the output of the previous layer
    - h_prev is the height of the previous layer
    - w_prev is the width of the previous layer
    - kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
    - kh is the kernel height
    - kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    - sh is the stride for the height
    - sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively
    Returns: the partial derivatives with respect to the previous layer
        (dA_prev)
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((m, h_prev, w_prev, c))
    for index in range(m):
        for k_index in range(c):
            for h in range(h_new):
                for w in range(w_new):
                    n_sh = h * sh
                    n_sw = w * sw
                    if mode == 'max':
                        pool = A_prev[index, n_sh: n_sh + kh,
                                      n_sw: n_sw + kw, k_index]
                        mask = np.where(pool == np.max(pool), 1, 0)
                    elif mode == 'avg':
                        mask = np.ones((kh, kw)) / (kh * kw)
                    dA_prev[index, n_sh: n_sh + kh,
                            n_sw: n_sw + kw, k_index] += (
                        mask * dA[index, h, w, k_index])
    return dA_prev
