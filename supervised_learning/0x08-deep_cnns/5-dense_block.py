#!/usr/bin/env python3
"""Dense Block"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number
        of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    Returns: The concatenated output of each layer
        within the Dense Block and the number of
        filters within the concatenated outputs,
        respectively
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    for layer in range(layers):
        B1 = K.layers.BatchNormalization(axis=3)(X)
        R1 = K.layers.Activation(activation)(B1)
        C1 = K.layers.Conv2D(
            filters=(4 * growth_rate),
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=kernel_initializer
        )(R1)
        B2 = K.layers.BatchNormalization(axis=3)(C1)
        R2 = K.layers.Activation(activation)(B2)
        C2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer
        )(R2)
        X = K.layers.concatenate([X, C2])
        nb_filters += growth_rate
    return X, nb_filters
