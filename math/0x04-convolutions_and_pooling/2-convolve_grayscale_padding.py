#!/usr/bin/env python3
"""Same Convolution"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images is a numpy.ndarray with shape (m, h, w) containing
        multiple grayscale images
    - m is the number of images
    - h is the height in pixels of the images
    - w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the
        kernel for the convolution
    - kh is the height of the kernel
    - kw is the width of the kernel
    padding is a tuple of (ph, pw)
    - ph is the padding for the height of the image
    - pw is the padding for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, w, h = images.shape[0], images.shape[2], images.shape[1]
    kw, kh = kernel.shape[1], kernel.shape[0]
    ph, pw = padding

    custom_h = h + (2 * ph) - kh + 1
    custom_w = w + (2 * pw) - kw + 1

    convolution = np.zeros((m, custom_h, custom_w))

    image_padded = np.pad(images, ((0, 0), (ph, ph),
                          (pw, pw)),
                          'constant', constant_values=0)

    for height in range(custom_h):
        for width in range(custom_w):
            output = np.sum(image_padded[:, height:height + kh,
                                         width: width + kw] * kernel,
                            axis=1).sum(axis=1)
            convolution[:, height, width] = output
    return convolution
