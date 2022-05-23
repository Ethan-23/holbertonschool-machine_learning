#!/usr/bin/env python3
"""Self Attention"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention (tf.keras.layers.Layer):
    """decode for machine translation"""

    def __init__(self, dm, h):
        """init function"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """call function"""
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = tf.reshape(q, (batch, -1, self.h, self.depth))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, (batch, -1, self.h, self.depth))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.reshape(v, (batch, -1, self.h, self.depth))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch, -1, self.dm))
        outputs = self.linear(attention)

        return outputs, weights
