#!/usr/bin/env python3
"""0. Initialize K-means"""

import sklearn.mixture


def gmm(X, k):
    """
    gmm Function
    """
    Gmm = sklearn.mixture.GaussianMixture(k)
    params = Gmm.fit(X)
    clss = Gmm.predict(X)
    return (params.weights_, params.means_,
            params.covariances_, clss, Gmm.bic(X))
