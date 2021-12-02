#!/usr/bin/env python3
"""5-across_the_planes"""

def add_matrices2D(mat1, mat2):
    """add_matrices2D"""
    new_mat = []
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    for i in range(len(mat1)):
        temp_mat = []
        for j in range(len(mat1[i])):
            temp_mat.append(mat1[i][j] + mat2[i][j])
        new_mat.append(temp_mat)
    return new_mat
