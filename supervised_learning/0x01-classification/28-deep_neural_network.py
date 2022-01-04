#!/usr/bin/env python3
"""17. Privatize DeepNeuralNetwork"""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """DeepNeuralNetwork Class"""

    def __init__(self, nx, layers, activation='sig'):
        """
        Init for DeepNeuralNetwork Class
        activation represents the type of activation function
            used in the hidden layers
        - sig represents a sigmoid activation
        - tanh represents a tanh activation
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
        if activation != "sig" or activation != "tanh":
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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

    @property
    def activation(self):
        """Getter for activation"""
        return self.__activation

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
            if i == self.__L - 1:
                t = np.exp(Z)
                self.cache[a] = (t / np.sum(t, axis=0, keepdims=True))
            else:
                if self.activation == "sig":
                    self.cache[a] = (1 / (1 + np.exp(-Z)))
                else:
                    self.cache[a] = np.tanh(Z)
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
        m = Y.shape[1]
        loss = np.sum(Y * np.log(A))
        cost = (1/m) * np.sum(-loss)
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
                bptemp = np.matmul(W_prev.transpose(), prev)
                if self.__activation == "sig":
                    bp["dz{}".format(i)] = bptemp * (cache[a] * (1 - cache[a]))
                else:
                    bp["dz{}".format(i)] = bptemp * (1 - cache[a] ** 2)
            dz = bp["dz{}".format(i)]
            dw = (1/m) * np.matmul(dz, cache[a_prev].transpose())
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            W_prev = self.weights[w]
            self.weights[w] = self.weights[w] - (alpha * dw)
            self.weights[b] = self.weights[b] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network
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
            A = self.cache["A{}".format(self.L)]
            if verbose and iteration % step == 0:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(iteration, cost))
            if graph and iteration % step == 0:
                values.append(self.cost(Y, A))
            self.gradient_descent(Y, self.cache, alpha)
        iteration += 1
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after {} iterations: {}".format(iteration, cost))
        if graph:
            values.append(self.cost(Y, A))
            y_values = np.asarray(values)
            plt.plot(x_values, y_values, 'b')
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        filename is the file to which the object should be saved
        """
        import pickle
        if type(filename) != str:
            return None
        if filename[-4:] != ".pkl":
            filename = filename[:] + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        filename is the file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn’t exist
        """
        import pickle
        try:
            with open(filename, 'rb') as f:
                loaded_object = pickle.load(f)
                return loaded_object
        except OSError:
            return None
