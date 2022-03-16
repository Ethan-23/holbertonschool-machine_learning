#!/usr/bin/env python3
"""Determinant"""


def determinant(matrix):
    """Calculates the Determinant of given matrix"""
    if type(matrix) != list or len(matrix) == 0 or type(matrix[0]) != list:
        raise TypeError("matrix must be a list of lists")
    if matrix==[[]]:
        return 1
    for i in matrix:
        if len(i) != len(matrix):
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
