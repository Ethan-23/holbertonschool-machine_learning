#!/usr/bin/env python3
"""Determinant"""


def determinant(matrix):
    """Calculates the Determinant of given matrix"""
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    elif len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if len(matrix) == 3:
        a = matrix[0][0] * (matrix[1][1] * matrix[2][2] -
                            matrix[1][2] * matrix[2][1])
        b = matrix[0][1] * (matrix[1][0] * matrix[2][2] -
                            matrix[1][2] * matrix[2][0])
        c = matrix[0][2] * (matrix[1][0] * matrix[2][1] -
                            matrix[1][1] * matrix[2][0])
        return a - b + c
