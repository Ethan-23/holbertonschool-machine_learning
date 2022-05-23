#!/usr/bin/env python3
"""Self Attention"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock (tf.keras.layers.Layer):
    """decode for machine translation"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """init function"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """call function"""
        output, _ = self.mha(x, x, x, mask)
        output = self.dropout1(output, training=training)
        output = self.layernorm1(x + output)

        dense_output = self.dense_hidden(output)
        ffn_output = self.dense_output(dense_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output + ffn_output)

        return output2
