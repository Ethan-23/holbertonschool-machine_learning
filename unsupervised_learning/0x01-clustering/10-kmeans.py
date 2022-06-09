#!/usr/bin/env python3
"""0. Initialize K-means"""

import sklearn.cluster


def kmeans(X, k):
    """
    kmeans Function
    """
    kmean = sklearn.cluster.KMeans(k)
    kmean.fit(X)
    return kmean.cluster_centers_, kmean.labels_
