#!/usr/bin/env python3
"""0. Initialize K-means"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    agglomerative
    """
    Z = scipy.cluster.hierarchy.linkage(X, method="ward")
    dendo = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    plt.show()
    return clss
