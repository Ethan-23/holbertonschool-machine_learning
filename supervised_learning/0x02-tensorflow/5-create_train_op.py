#!/usr/bin/env python3
"""5. Train_Op"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent
    """
    train = tf.train.GradientDescentOptimizer(
        learning_rate=alpha,
    )
    return train.minimize(loss)
