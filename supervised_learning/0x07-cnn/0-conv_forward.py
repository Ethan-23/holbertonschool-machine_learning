#!/usr/bin/env python3
"""Convolutional Forward Prop"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
    - m is the number of examples
    - h_prev is the height of the previous layer
    - w_prev is the width of the previous layer
    - c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        W for the convolution
    - kh is the filter height
    - kw is the filter width
    - c_prev is the number of channels in the previous layer
    - c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of
        padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    - sh is the stride for the height
    - sw is the stride for the width
    Returns: the output of the convolutional layer
    """
    m, h, w, c = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        return

    custom_h = ((h + (2 * ph) - kh)//sh) + 1
    custom_w = ((w + (2 * pw) - kw)//sw) + 1

    convolution = np.zeros((m, custom_h, custom_w, nc))

    image_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                          (pw, pw), (0, 0)),
                          'constant', constant_values=0)
    for k_num in range(nc):
        kernel_index = W[:, :, :, k_num]
        i = 0
        for height in range(0, (h + (2 * ph) - kh + 1), sh):
            j = 0
            for width in range(0, (w + (2 * pw) - kw + 1), sw):
                output = np.sum(image_padded[:, height:height + kh,
                                             width: width + kw,
                                             :] * kernel_index,
                                axis=1).sum(axis=1).sum(axis=1)
                output += b[0, 0, 0, k_num]
                convolution[:, i, j, k_num] = activation(output)
                j += 1
            i += 1
    return convolution
