#!/usr/bin/env python3
"""7. Upgrade Train Neuron"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Neuron Class"""

    def __init__(self, nx):
        """
        Init for Neuron Class
        nx: is the number of input features to the neuron
        W: The weights vector for the neuron. Upon instantiation,
            it should be initialized using a random normal distribution.
        b: The bias for the neuron. Upon instantiation, it should be
            initialized to 0.
        A: The activated output of the neuron (prediction). Upon
            instantiation, it should be initialized to 0.
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W"""
        return self.__W

    @property
    def b(self):
        """Getter for b"""
        return self.__b

    @property
    def A(self):
        """Getter for A"""
        return self.__A

    def sigmoid(self, z):
        """Sigmoid Function"""
        return 1/(1 + np.exp(-z))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example
        alpha is the learning rate
        """
        m = Y.shape[1]
        dw = (1/m) * (np.matmul(X, (A - Y).transpose()).transpose())
        db = (1/m) * np.sum(A - Y)
        self.__W = self.W - (alpha * dw)
        self.__b = self.b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
        iterations is the number of iterations to train over
        alpha is the learning rate
        verbose is a boolean that defines whether or not to print information
            about the training. If True, print Cost after {iteration}
            iterations: {cost} every step iterations:
        Include data from the 0th and last iteration
        graph is a boolean that defines whether or not to graph information
            about the training once the training has completed
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
                cost = self.cost(Y, self.A)
                print("Cost after {} iterations: {}".format(iteration, cost))
            if graph and iteration % step == 0:
                values.append(self.cost(Y, self.A))
            self.gradient_descent(X, Y, self.A, alpha)
        iteration += 1
        if verbose:
            cost = self.cost(Y, self.A)
            print("Cost after {} iterations: {}".format(iteration, cost))
        if graph:
            values.append(self.cost(Y, self.A))
            y_values = np.asarray(values)
            plt.plot(x_values, y_values, 'b')
        return self.evaluate(X, Y)
