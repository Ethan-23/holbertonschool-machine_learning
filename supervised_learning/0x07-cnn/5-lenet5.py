#!/usr/bin/env python3
"""Convolutional Back Prop"""

import tensorflow.keras as K


def lenet5(X):
    """
    X is a K.Input of shape (m, 28, 28, 1) containing the input
        images for the network
    - m is the number of images
    Returns: a K.Model compiled to use Adam optimization
        (with default hyperparameters) and accuracy metrics
    """
    kernel_initializer = K.initializers.he_normal()
    activation = K.activations.relu
    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_1 = C1(X)

    M2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
    output_2 = M2(output_1)

    C3 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_3 = C3(output_2)

    M4 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
    output_4 = M4(output_3)
    output_41 = K.layers.Flatten()(output_4)

    F5 = K.layers.Dense(
        120,
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_5 = F5(output_41)

    F6 = K.layers.Dense(
        84,
        activation=activation,
        kernel_initializer=kernel_initializer
    )
    output_6 = F6(output_5)

    F7 = K.layers.Dense(
        10,
        kernel_initializer=kernel_initializer
    )
    output_7 = F7(output_6)

    softmax = K.layers.Softmax()(output_7)
    model = K.Model(
        inputs=X,
        outputs=softmax
    )
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
