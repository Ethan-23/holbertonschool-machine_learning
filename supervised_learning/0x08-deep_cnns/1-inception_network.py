#!/usr/bin/env python3
"""Inception Block"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Returns: the keras model
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    img_input = K.Input(shape=(224, 224, 3))
    C1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer
    )(img_input)
    M1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(C1)
    C2 = K.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer
    )(M1)
    C3 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer
    )(C2)
    M2 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(C3)
    I1 = inception_block(M2, [64, 96, 128, 16, 32, 32])
    I2 = inception_block(I1, [128, 128, 192, 32, 96, 64])
    M3 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(I2)
    I3 = inception_block(M3, [192, 96, 208, 16, 48, 64])
    I4 = inception_block(I3, [160, 112, 224, 24, 64, 64])
    I5 = inception_block(I4, [128, 128, 256, 24, 64, 64])
    I6 = inception_block(I5, [112, 144, 288, 32, 64, 64])
    I7 = inception_block(I6, [256, 160, 320, 32, 128, 128])
    M4 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(I7)
    I8 = inception_block(M4, [256, 160, 320, 32, 128, 128])
    I9 = inception_block(I8, [384, 192, 384, 48, 128, 128])
    A1 = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(I9)
    D1 = K.layers.Dropout(rate=0.4)(A1)
    output = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=kernel_initializer
    )(D1)
    model = K.Model(inputs=img_input, outputs=output)
    return model
