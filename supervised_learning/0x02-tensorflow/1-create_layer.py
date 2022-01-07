#!/usr/bin/env python3
"""1-create_layer.py"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    Returns: the tensor output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernel_initializer=initializer
    )
    return (layer(prev))
