#!/usr/bin/env python3
"""RNN Encoder"""

import numpy as np
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """encode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """public instance attributes"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.units = units
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN
           cell to a tensor of zeros"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """call"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
