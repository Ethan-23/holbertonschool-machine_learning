#!/usr/bin/env python3
"""Sequential"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the
        network
    activations is a list containing the activation functions used for
        each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    """

    inputs = K.Input(shape=(nx, ))
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            outputs = K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2)(inputs)
        else:
            dropout = K.layers.Dropout(1-keep_prob)(outputs)
            outputs = K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2)(dropout)
    return K.models.Model(inputs=inputs, outputs=outputs)
