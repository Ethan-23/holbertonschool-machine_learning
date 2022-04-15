#!/usr/bin/env python3
"""3. Initialize Bayesian Optimization"""

from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """class"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing the
            inputs already sampled with the black-box afunction
        Y_init is a numpy.ndarray of shape (t, 1) representing the
            outputs of the black-box function for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds
            of the space in which to look for the optimal point
        ac_samples is the number of samples that should be analyzed
            during acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
            black-box function
        xsi is the exploration-exploitation factor for acquisition
        minimize is a bool determining whether optimization should be
            performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples)[:, None]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        """
        fs, _ = self.gp.predict(self.gp.X)
        next_fs, vars = self.gp.predict(self.X_s)
        opt = np.min(fs)
        improves = opt - next_fs - self.xsi
        if not self.minimize:
            improve = -improves
        Z = improves / vars
        eis = improves * norm.cdf(Z) + vars * norm.pdf(Z)
        return self.X_s[np.argmax(eis)], eis
