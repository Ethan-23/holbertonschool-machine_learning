#!/usr/bin/env python3
"""Valid Convolution"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
        containing multiple images
    - m is the number of images
    - h is the height in pixels of the images
    - w is the width in pixels of the images
    - c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the
        kernel shape for the pooling
    - kh is the height of the kernel
    - kw is the width of the kernel
    stride is a tuple of (sh, sw)
    - sh is the stride for the height of the image
    - sw is the stride for the width of the image
    mode indicates the type of pooling
    - max indicates max pooling
    - avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
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
                output = np.max(images[:, height:height + kh,
                                       width:width + kw, :],
                                axis=(1, 2))
            elif mode == 'avg':
                output = np.average(images[:, height:height + kh,
                                           width:width + kw, :],
                                    axis=(1, 2))
            else:
                pass
            pooled[:, i, j, :] = output
            j += 1
        i += 1
    return pooled
