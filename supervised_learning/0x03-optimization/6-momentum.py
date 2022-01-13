#!/usr/bin/env python3
"""6-mometum"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the momentum weight
    Returns: the momentum optimization operation
    """
    op = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return op
