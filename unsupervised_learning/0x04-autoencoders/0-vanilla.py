#!/usr/bin/env python3
"""0x04. Autoencoders"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """autoencoder"""
    inputs = keras.Input((input_dims,))
    prev = inputs
    for node_count in range(len(hidden_layers)):
        prev = keras.layers.Dense(hidden_layers[node_count], 'relu')(prev)

    encoder_layers = keras.layers.Dense(latent_dims, 'relu')(prev)
    encoder = keras.Model(inputs, encoder_layers)

    dec_input = keras.Input((latent_dims,))
    prev = dec_input
    for node_count in reversed(range(len(hidden_layers))):
        prev = keras.layers.Dense(hidden_layers[node_count], 'relu')(prev)

    decoder_layers = keras.layers.Dense(input_dims, 'sigmoid')(prev)
    decoder = keras.Model(dec_input, decoder_layers)

    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return (encoder, decoder, auto)
