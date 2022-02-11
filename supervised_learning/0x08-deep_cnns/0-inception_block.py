#!/usr/bin/env python3
"""Inception Block"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R,
        F5, FPP, respectively:
    - F1 is the number of filters in the 1x1 convolution
    - F3R is the number of filters in the 1x1 convolution
        before the 3x3 convolution
    - F3 is the number of filters in the 3x3 convolution
    - F5R is the number of filters in the 1x1 convolution
        before the 5x5 convolution
    - F5 is the number of filters in the 5x5 convolution
    - FPP is the number of filters in the 1x1 convolution
        after the max pooling (Note : The output shape after
        the max pooling layer is outputshape =
        math.floor((inputshape - 1) / strides) + 1)
    Returns: the concatenated output of the inception block
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    F1, F3R, F3, F5R, F5, FPP = filters
    # 1x1 convoluition
    C1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_1 = C1(A_prev)
    # 1x1 convoluition -> 3x3 convoluition
    C2 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_2 = C2(A_prev)
    C3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_3 = C3(output_2)
    # 1x1 convoluition -> 5x5 convoluition
    C4 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_4 = C4(A_prev)
    C5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_5 = C5(output_4)
    # 3x3 max pooling -> 1x1 convoluition
    M1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )
    output_6 = M1(A_prev)
    C6 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_7 = C6(output_6)

    return(K.layers.concatenate([output_1, output_3, output_5, output_7]))
