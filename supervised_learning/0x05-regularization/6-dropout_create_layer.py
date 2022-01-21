#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function that should be used on the layer
    keep_prob is the probability that a node will be kept
    Returns: the output of the new layer
    """
    kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                               mode="fan_avg"
                                                               )
    kernel_regularizer = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer
                            )
    return layer(prev)
