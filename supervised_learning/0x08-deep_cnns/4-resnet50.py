#!/usr/bin/env python3
"""Inception Network"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds an inception network
    Returns: the keras model
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    img_input = K.layers.Input((224, 224, 3))
    C1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=(2, 2),
        kernel_initializer=kernel_initializer
    )(img_input)
    B1 = K.layers.BatchNormalization(axis=3)(C1)
    R1 = K.layers.Activation(activation)(B1)
    M1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(R1)
    P1 = projection_block(M1, [64, 64, 256], s=1)
    I1 = identity_block(P1, [64, 64, 256])
    I2 = identity_block(I1, [64, 64, 256])

    P2 = projection_block(I2, [128, 128, 512], s=2)
    I3 = identity_block(P2, [128, 128, 512])
    I4 = identity_block(I3, [128, 128, 512])
    I5 = identity_block(I4, [128, 128, 512])

    P3 = projection_block(I5, [256, 256, 1024], s=2)
    I6 = identity_block(P3, [256, 256, 1024])
    I7 = identity_block(I6, [256, 256, 1024])
    I8 = identity_block(I7, [256, 256, 1024])
    I9 = identity_block(I8, [256, 256, 1024])
    I10 = identity_block(I9, [256, 256, 1024])

    P4 = projection_block(I10, [512, 512, 2048], s=2)
    I11 = identity_block(P4, [512, 512, 2048])
    I12 = identity_block(I11, [512, 512, 2048])

    A = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(I12)

    output = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=kernel_initializer
    )(A)
    model = K.Model(inputs=img_input, outputs=output)
    return model
