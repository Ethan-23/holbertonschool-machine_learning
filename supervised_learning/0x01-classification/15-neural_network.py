#!/usr/bin/env python3
"""9. Privatize NeuralNetwork"""

import numpy as np
import matplotlib.pyplot as plt


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
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        pred = np.where(A2 >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A1 is the output of the hidden layer
        A2 is the predicted output
        alpha is the learning rate
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1/m) * np.matmul(dz2, A1.transpose())
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.W2.transpose(), dz2) * (A1 * (1 - A1))
        dw1 = (1/m) * np.matmul(dz1, X.transpose())
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.W1 - (alpha * dw1)
        self.__b1 = self.b1 - (alpha * db1)
        self.__W2 = self.W2 - (alpha * dw2)
        self.__b2 = self.b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
        iterations is the number of iterations to train over
        alpha is the learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            x_values = np.arange(0, iterations + 1, step)
            values = []
        for iteration in range(iterations):
            self.forward_prop(X)
            if verbose and iteration % step == 0:
                cost = self.cost(Y, self.A2)
                print("Cost after {} iterations: {}".format(iteration, cost))
            if graph and iteration % step == 0:
                values.append(self.cost(Y, self.A2))
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        iteration += 1
        if verbose:
            cost = self.cost(Y, self.A2)
            print("Cost after {} iterations: {}".format(iteration, cost))
        if graph:
            values.append(self.cost(Y, self.A2))
            y_values = np.asarray(values)
            plt.plot(x_values, y_values, 'b')
        return self.evaluate(X, Y)
