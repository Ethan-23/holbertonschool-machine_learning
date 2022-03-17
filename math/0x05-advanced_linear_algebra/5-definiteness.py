#!/usr/bin/env python3
"""Determinant"""

import numpy as np


def definiteness(matrix):
    """definiteness"""
    if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2:
        return None
    for i in matrix:
        if len(matrix) != len(i):
            return None
    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"
    if np.all(np.linalg.eigvals(matrix) >= 0):
        return "Positive semi-definite"
    if np.all(np.linalg.eigvals(matrix) < 0):
        return "Negative definite"
    if np.all(np.linalg.eigvals(matrix) <= 0):
        return "Negative semi-definite"
    low = np.all(np.linalg.eigvals(matrix) <= 0)
    high = np.all(np.linalg.eigvals(matrix) >= 0)
    if not low and not high:
        return "Indefinite"
    return None
