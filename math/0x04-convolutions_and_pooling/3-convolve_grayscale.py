#!/usr/bin/env python3
"""Same Convolution"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    - ‘same’, performs a same convolution
    - ‘valid’, performs a valid convolution
    - if a tuple:
     - ph is the padding for the height of the image
     - pw is the padding for the width of the image
    stride is a tuple of (sh, sw)
    - sh is the stride for the height of the image
    - sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, w, h = images.shape[0], images.shape[2], images.shape[1]
    kw, kh = kernel.shape[1], kernel.shape[0]
    sh, sw = stride

    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    custom_h = ((h + (2 * ph) - kh)//sh) + 1
    custom_w = ((w + (2 * pw) - kw)//sw) + 1

    convolution = np.zeros((m, custom_h, custom_w))

    image_padded = np.pad(images, ((0, 0), (ph, ph),
                          (pw, pw)),
                          'constant', constant_values=0)
    i = 0
    for height in range(0, (h + (2 * ph) - kh + 1), sh):
        j = 0
        for width in range(0, (w + (2 * pw) - kw + 1), sw):
            output = np.sum(image_padded[:, height:height + kh,
                                         width: width + kw] * kernel,
                            axis=1).sum(axis=1)
            convolution[:, i, j] = output
            j += 1
        i += 1
    return convolution
