#!/usr/bin/env python3
"""Dense Block"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the
        number of filters in X
    compression is the compression factor for
        the transition layer
    Returns: The output of the transition
        layer and the number of filters within
        the output, respectively
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    B1 = K.layers.BatchNormalization(axis=3)(X)
    R1 = K.layers.Activation(activation)(B1)
    nb_filters *= compression
    nb_filters = int(nb_filters)
    C1 = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=kernel_initializer
    )(R1)
    A1 = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(C1)
    return A1, nb_filters
