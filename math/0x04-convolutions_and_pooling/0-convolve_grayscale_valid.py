#!/usr/bin/env python3
"""Valid Convolution"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w) containing
        multiple grayscale images
    - m is the number of images
    - h is the height in pixels of the images
    - w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing
        the kernel for the convolution
    - kh is the height of the kernel
    - kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m, w, h = images.shape[0], images.shape[2], images.shape[1]
    kw, kh = kernel.shape[1], kernel.shape[0]
    convolution = np.zeros((m, h - kh + 1, w - kw + 1))
    for height in range(h - kh + 1):
        for width in range(w - kw + 1):
            output = np.sum(images[:, height:height + kh,
                                   width: width + kw] * kernel,
                            axis=1).sum(axis=1)
            convolution[:, height, width] = output
    return convolution
