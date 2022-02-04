#!/usr/bin/env python3
"""Pooling Forward Prop"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing
        the output of the previous layer
    - m is the number of examples
    - h_prev is the height of the previous layer
    - w_prev is the width of the previous layer
    - c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the
        kernel for the pooling
    - kh is the kernel height
    - kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the
        pooling
    - sh is the stride for the height
    - sw is the stride for the width
    mode is a string containing either max or avg, indicating
        whether to perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = ((h - kh) // sh) + 1
    pw = ((w - kw) // sw) + 1
    pooled = np.zeros((m, ph, pw, c))
    i = 0
    for height in range(0, (h - kh + 1), sh):
        j = 0
        for width in range(0, (w - kw + 1), sw):
            if mode == 'max':
                output = np.max(A_prev[:, height:height + kh,
                                       width:width + kw, :],
                                axis=(1, 2))
            elif mode == 'avg':
                output = np.average(A_prev[:, height:height + kh,
                                           width:width + kw, :],
                                    axis=(1, 2))
            else:
                pass
            pooled[:, i, j, :] = output
            j += 1
        i += 1
    return pooled
