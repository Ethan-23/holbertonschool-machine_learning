#!/usr/bin/env python3
"""Initialize Poisson"""


class Poisson:
    """Poisson class"""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Init for poisson class"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Poisson PMF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        fact = 1
        for i in range(1, k+1):
            fact *= i
        return (self.lambtha ** k * self.e ** -self.lambtha) / fact

    def cdf(self, k):
        """Poisson PMF"""
        if type(k) is not int:
            k = int(k)
        if k <= 0:
            return 0
        num = 0
        while k >= 0:
            num += self.pmf(k)
            k -= 1
        return num
