#!/usr/bin/env python3
"""Self Attention"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer (tf.keras.Model):
    """decode for machine translation"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """init function"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """call function"""
        encoder = self.encoder(inputs, training, encoder_mask)
        decoder = self.decoder(target, encoder, training,
                               look_ahead_mask, decoder_mask)
        return self.linear(decoder)
