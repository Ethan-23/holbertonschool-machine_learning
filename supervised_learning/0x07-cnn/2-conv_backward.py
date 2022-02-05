#!/usr/bin/env python3
"""Convolutional Back Prop"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer
    - m is the number of examples
    - h_new is the height of the output
    - w_new is the width of the output
    - c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
    - h_prev is the height of the previous layer
    - w_prev is the width of the previous layer
    - c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
    - kh is the filter height
    - kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    padding is a string that is either same or valid, indicating the type
        of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    - sh is the stride for the height
    - sw is the stride for the width
    Returns: the partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_new, w_new, c_new = dZ.shape
    sh, sw = stride

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = ((((h_prev - 1) * sh) + kh - h_prev) // 2) + 1
        pw = ((((w_prev - 1) * sw) + kw - w_prev) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        return

    image_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                          (pw, pw), (0, 0)),
                          'constant', constant_values=0)

    dA_prev = np.zeros((m, h_prev + (2 * ph), w_prev + (2 * pw), c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))

    for index in range(m):
        for k_index in range(c_new):
            for h in range(h_new):
                for w in range(w_new):
                    n_sh = h * sh
                    n_sw = w * sw
                    dA_prev[index, n_sh:n_sh + kh, n_sw:n_sw + kw, :] += (
                        dZ[index, h, w, k_index] * W[:, :, :, k_index])
                    dW[:, :, :, k_index] += (
                        image_padded[index, n_sh:n_sh + kh,
                                     n_sw:n_sw + kw, :] *
                        dZ[index, h, w, k_index])
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dW, db
