#!/usr/bin/env python3
"""16. DeepNeuralNetwork"""

import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork Class"""

    def __init__(self, nx, layers):
        """
        Init for DeepNeuralNetwork Class
        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        layers is a list representing the number of nodes in each
            layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        pre = nx
        for i in range(len(layers)):
            if layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(i + 1)
            b = "b{}".format(i + 1)
            self.weights[w] = np.random.randn(layers[i], pre) * np.sqrt(2/pre)
            self.weights[b] = np.zeros((layers[i], 1))
            pre = layers[i]
