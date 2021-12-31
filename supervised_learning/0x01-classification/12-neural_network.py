#!/usr/bin/env python3
"""9. Privatize NeuralNetwork"""

import numpy as np


class NeuralNetwork:
    """NeuralNetwork Class"""

    def __init__(self, nx, nodes):
        """
        Init for Neuron Class
        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W2"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b2"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A2"""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2"""
        return self.__A2

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
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.A1, self.A2

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
        Evaluates the neuron’s predictions
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        Returns the neuron’s prediction and the cost of the network,
            respectively
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost
