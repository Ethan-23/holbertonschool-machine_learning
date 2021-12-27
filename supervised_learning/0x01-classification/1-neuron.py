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
        self.__W = np.random.normal(size=(1,nx))
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
