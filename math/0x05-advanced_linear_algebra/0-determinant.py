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
    total = 0
    for i in list(range(len(matrix))):
        tempMatrix = matrix
        tempMatrix = tempMatrix[1:]
        height = len(tempMatrix)
        for j in range(height):
            tempMatrix[j] = tempMatrix[j][0:i] + tempMatrix[j][i+1:]
        sign = (-1) ** (i % 2)
        sub_det = determinant(tempMatrix)
        total += sign * matrix[0][i] * sub_det
    return total
