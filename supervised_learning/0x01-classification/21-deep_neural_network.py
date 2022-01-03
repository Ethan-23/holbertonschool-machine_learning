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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        Returns the cost
        """
        loss = -(Y*np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.sum(1/Y.shape[1] * loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
        Returns the neuron’s prediction and the cost of the network,
            respectively
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Y is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data
        cache is a dictionary containing all the intermediary
            values of the network
        alpha is the learning rate
        """
        m = Y.shape[1]
        bp = {}
        for i in range(self.L, 0, -1):
            a = "A{}".format(i)
            a_prev = "A{}".format(i-1)
            w = "W{}".format(i)
            b = "b{}".format(i)
            if self.L == i:
                bp["dz{}".format(i)] = cache[a] - Y
            else:
                prev = bp["dz{}".format(i + 1)]
                bp["dz{}".format(i)] = np.matmul(W_prev.transpose(),
                                                 prev) * (cache[a] *
                                                          (1 - cache[a]))
            dz = bp["dz{}".format(i)]
            dw = (1/m) * np.matmul(dz, cache[a_prev].transpose())
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            W_prev = self.weights[w]
            self.weights[w] = self.weights[w] - (alpha * dw)
            self.weights[b] = self.weights[b] - (alpha * db)
