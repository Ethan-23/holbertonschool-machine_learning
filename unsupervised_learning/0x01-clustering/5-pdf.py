#!/usr/bin/env python3
"""0. Initialize K-means"""

import numpy as np


def pdf(X, m, S):
    """
    pdf
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or (d, d) != S.shape:
        return None
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    a = 1 / np.sqrt((((2 * np.pi) ** (d) * det)))
    inv = np.einsum('...k,kl,...l->...', (X - m), inv, (X - m))
    b = np.exp(-(1 / 2) * inv)
    pdf = a * b
    pdf = np.where(pdf >= 1e-300, pdf, 1e-300)
    return pdf
