#!/usr/bin/env python3
"""Same Convolution"""

import numpy as np


def convolve_grayscale_same(images, kernel):
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

    padding_along_height = max((h - 1) * 1 + kh - h, 0)
    padding_along_width = max((w - 1) * 1 + kw - w, 0)
    pad_top = padding_along_height // 2
    pad_bot = padding_along_height - pad_top
    pad_left = padding_along_width // 2
    pad_right = padding_along_width - pad_left

    convolution = np.zeros((m, h, w))

    image_padded = np.pad(images, ((0, 0), (pad_top, pad_bot),
                          (pad_left, pad_right)),
                          'constant', constant_values=0)

    for height in range(h):
        for width in range(w):
            output = np.sum(image_padded[:, height:height + kh,
                                         width: width + kw] * kernel,
                            axis=1).sum(axis=1)
            convolution[:, height, width] = output
    return convolution
