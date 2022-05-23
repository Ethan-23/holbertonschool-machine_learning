#!/usr/bin/env python3
"""Self Attention"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """decode for machine translation"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """init function"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """call function"""
        output, _ = self.mha1(x, x, x, look_ahead_mask)
        output = self.dropout1(output, training=training)
        output = self.layernorm1(x + output)

        output2, _ = self.mha2(output, encoder_output,
                               encoder_output, padding_mask)
        output2 = self.dropout2(output2, training=training)
        output2 = self.layernorm2(output + output2)

        dense_output = self.dense_hidden(output2)
        ffn_output = self.dense_output(dense_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        output3 = self.layernorm2(output2 + ffn_output)

        return output3
