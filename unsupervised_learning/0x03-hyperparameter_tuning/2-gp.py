#!/usr/bin/env python3
"""0-gp.py"""


import numpy as np


class GaussianProcess:
    """Class"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
            already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
            of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of
            the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        Returns: the covariance kernel matrix as a
            numpy.ndarray of shape (m, n)
        """
        X1_sum = np.sum(X1 ** 2, 1).reshape(-1, 1)
        X2_sum = np.sum(X2 ** 2, 1)
        sqdist = X1_sum + X2_sum - 2 * np.matmul(X1, X2.T)
        cov = (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)
        return cov

    def predict(self, X_s):
        """
        X_s is a numpy.ndarray of shape (s, 1) containing all of the points
            whose mean and standard deviation should be calculated
        s is the number of sample points
        Returns: mu, sigma
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_inv = np.linalg.inv(K)
        K_ss = self.kernel(X_s, X_s)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu = mu_s.reshape(-1)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)
        return mu, sigma

    def update(self, X_new, Y_new):
        """
        X_new is a numpy.ndarray of shape (1,) that represents the new
            sample point
        Y_new is a numpy.ndarray of shape (1,) that represents the new
            sample function value
        Updates the public instance attributes X, Y, and K
        """
        self.X = np.append(self.X, X_new)[:, np.newaxis]
        self.Y = np.append(self.Y, Y_new)[:, np.newaxis]
        self.K = self.kernel(self.X, self.X)
        