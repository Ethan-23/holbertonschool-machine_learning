#!/usr/bin/env python3
"""0x04. Autoencoders"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """autoencoder"""
    encoder_inputs = keras.Input(shape=(input_dims))
    encoder_value = encoder_inputs
    for node_count in range(len(filters)):
        layer = keras.layers.Conv2D(filters[node_count],
                                    activation='relu',
                                    kernel_size=(3, 3),
                                    padding='same')
        encoder_value = layer(encoder_value)
        layer = keras.layers.MaxPooling2D((2, 2), padding='same')
        encoder_value = layer(encoder_value)
    encoder = keras.Model(inputs=encoder_inputs, outputs=encoder_value)

    decoder_inputs = keras.Input(shape=(latent_dims))
    decoder_value = decoder_inputs
    for node_count in reversed(range(len(filters) - 1)):
        layer = keras.layers.Conv2D(filters[node_count],
                                    activation='relu',
                                    kernel_size=(3, 3),
                                    padding='same')
        decoder_value = layer(decoder_value)
        layer = keras.layers.UpSampling2D((2, 2))
        decoder_value = layer(decoder_value)
    layer = keras.layers.Conv2D(filters[0],
                                activation='relu',
                                kernel_size=(3, 3),
                                padding='valid')
    decoder_value = layer(decoder_value)
    layer = keras.layers.UpSampling2D((2, 2))
    decoder_value = layer(decoder_value)
    layer = keras.layers.Conv2D(input_dims[2],
                                activation='sigmoid',
                                kernel_size=(3, 3),
                                padding='same')
    decoder_outputs = layer(decoder_value)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    inputs = encoder_inputs
    auto = keras.Model(inputs=encoder_inputs, outputs=decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return (encoder, decoder, auto)
