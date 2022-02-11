#!/usr/bin/env python3
"""Projection Block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
    - F11 is the number of filters in the first 1x1 convolution
    - F3 is the number of filters in the 3x3 convolution
    - F12 is the number of filters in the second 1x1 convolution
    s is the stride of the first convolution in both the main
        path and the shortcut connection
    Returns: the activated output of the identity block
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    F11, F3, F12 = filters
    C1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        strides=s,
        kernel_initializer=kernel_initializer
    )(A_prev)
    B1 = K.layers.BatchNormalization(axis=3)(C1)
    R1 = K.layers.Activation(activation)(B1)
    C2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=kernel_initializer
    )(R1)
    B2 = K.layers.BatchNormalization(axis=3)(C2)
    R2 = K.layers.Activation(activation)(B2)
    C3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=kernel_initializer
    )(R2)
    B3 = K.layers.BatchNormalization(axis=3)(C3)
    C4 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        strides=s,
        kernel_initializer=kernel_initializer
    )(A_prev)
    B4 = K.layers.BatchNormalization(axis=3)(C4)
    Add = K.layers.Add()([B3, B4])
    output = K.layers.Activation(activation)(Add)
    return(output)
