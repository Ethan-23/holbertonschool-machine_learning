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

    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx, ),
                                     activation=activations[i],
                                     kernel_regularizer=L2,
                                     name='dense'))
        else:
            model.add(K.layers.Dropout(1-keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2,
                                     name='dense_' + str(i)))
    return model
