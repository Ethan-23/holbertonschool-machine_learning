#!/usr/bin/env python3
"""Initialize Exponential"""


class Normal:
    """Exponential class"""

    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """Init for Exponential class"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            deviations = [(x - self.mean) ** 2 for x in data]
            variance = sum(deviations) / len(data)
            self.stddev = variance ** 0.5

        

    def z_score(self, x):
        """z_score function"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """x_value function"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Normal pdf"""

    def cdf(self, x):
        """Normal cdf"""