#!/usr/bin/env python3
"""Convolutional Back Prop"""

import numpy as np
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
        images for the network
    m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot
        labels for the network
    Returns:
    - a tensor for the softmax activated output
    - a training operation that utilizes Adam optimization
        (with default hyperparameters)
    - a tensor for the loss of the netowrk
    - a tensor for the accuracy of the network
    """
    kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    C1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer
    )
    output_1 = C1(x)

    M2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
    output_2 = M2(output_1)

    C3 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer
    )
    output_3 = C3(output_2)

    M4 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
    output_4 = M4(output_3)
    output_41 = tf.layers.Flatten()(output_4)

    F5 = tf.layers.Dense(
        120,
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer
    )
    output_5 = F5(output_41)

    F6 = tf.layers.Dense(
        84,
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer
    )
    output_6 = F6(output_5)

    F7 = tf.layers.Dense(
        10,
        kernel_initializer=kernel_initializer
    )
    output_7 = F7(output_6)

    softmax = tf.nn.softmax(output_7)
    loss = tf.losses.softmax_cross_entropy(
        y,
        logits=output_7
    )
    op = tf.train.AdamOptimizer().minimize(loss)
    correct = tf.equal(tf.argmax(output_7, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    return softmax, op, loss, accuracy
