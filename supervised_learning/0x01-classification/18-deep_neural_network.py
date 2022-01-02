#!/usr/bin/env python3
"""17. Privatize DeepNeuralNetwork"""

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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        pre = nx
        for i in range(len(layers)):
            if layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(i + 1)
            b = "b{}".format(i + 1)
            self.weights[w] = np.random.randn(layers[i], pre) * np.sqrt(2/pre)
            self.weights[b] = np.zeros((layers[i], 1))
            pre = layers[i]

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def sigmoid(self, z):
        """Sigmoid Function"""
        return 1/(1 + np.exp(-z))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        """
        prev = X
        for i in range(self.L):
            self.cache["A0"] = X
            a = "A{}".format(i + 1)
            w = "W{}".format(i + 1)
            b = "b{}".format(i + 1)
            Z = np.matmul(self.weights[w], prev) + self.weights[b]
            self.cache[a] = self.sigmoid(Z)
            prev = self.cache[a]
        return self.cache[a], self.cache
