#!/usr/bin/env python3

import numpy as np


class Neuron:
    """Neuron Class"""

    def __init__(self, nx):
        """
        Init for Neuron Class
        nx: is the number of input features to the neuron
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
