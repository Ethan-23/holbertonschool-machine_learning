#!/usr/bin/env python3
"""Positional Encoding"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """sdp_attention"""
    matmul = tf.matmul(Q, K, transpose_b=True)
    scale = tf.cast(tf.shape(K)[-1], tf.float32)
    scale = matmul / tf.math.sqrt(scale)
    if mask is not None:
        scale += (mask * -1e9)
    weights = tf.nn.softmax(scale, axis=-1)
    outputs = tf.matmul(weights, V)
    return outputs, weights
