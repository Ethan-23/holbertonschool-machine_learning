#!/usr/bin/env python3
"""3-flip_me_over"""

def matrix_transpose(matrix):
    """Transpose Matrix"""
    new_mat = []
    for i in range(len(matrix[0])):
        temp_mat = []
        for j in range(len(matrix)):
            temp_mat.append(matrix[j][i])
        new_mat.append(temp_mat)
    return new_mat