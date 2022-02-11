#!/usr/bin/env python3
"""DenseNet-121"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate is the growth rate
    compression is the compression factor
    Returns: the keras model
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    img_input = K.layers.Input((224, 224, 3))
    B1 = K.layers.BatchNormalization(axis=3)(img_input)
    R1 = K.layers.Activation(activation)(B1)
    C1 = K.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            padding='same',
            strides=(2, 2),
            kernel_initializer=kernel_initializer
        )(R1)
    M1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(C1)
    D1, nb_filters = dense_block(M1, 64, growth_rate, 6)
    T1, nb_filters = transition_layer(D1, nb_filters, compression)
    D2, nb_filters = dense_block(T1, nb_filters, growth_rate, 12)
    T2, nb_filters = transition_layer(D2, nb_filters, compression)
    D3, nb_filters = dense_block(T2, nb_filters, growth_rate, 24)
    T3, nb_filters = transition_layer(D3, nb_filters, compression)
    D4, nb_filters = dense_block(T3, nb_filters, growth_rate, 16)

    A1 = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(D4)

    output = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=kernel_initializer
    )(A1)

    model = K.Model(inputs=img_input, outputs=output)
    return model
