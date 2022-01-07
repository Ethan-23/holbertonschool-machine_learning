#!/usr/bin/env python3
"""4. Loss"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        y,
        logits=y_pred
    )
    return loss
