#!/usr/bin/env python3
"""Initialize Binomial"""


class Binomial:
    """Binomial class"""

    e = 2.7182818285

    def __init__(self, data=None, n=1, p=0.5):
        """Init for Binomial class"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = [(i - mean) ** 2 for i in data]
            self.p = 1 - ((sum(variance) / len(data)) / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """Binomial PMF"""
        if type(k) is not int:
            k = int(k)
        if k <= 0 or k >= self.n:
            return 0
        nfact = 1
        kfact = 1
        bfact = 1
        for num1 in range(1, k+1):
            kfact *= num1
        for num2 in range(1, self.n+1):
            nfact *= num2
        for num3 in range(1, (self.n - k)+1):
            bfact *= num3
        factorials = nfact / (kfact * bfact)
        return factorials * self.p ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """Binomial PMF"""
        if type(k) is not int:
            k = int(k)
        if k <= 0 or k >= self.n:
            return 0
        num = 0
        while k >= 0:
            num += self.pmf(k)
            k -= 1
        return num
