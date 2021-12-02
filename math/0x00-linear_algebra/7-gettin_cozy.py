#!/usr/bin/env python3
"""7-getting_cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """cat_matrices2D"""
    new_mat = []
    for i in range(len(mat1)):
        new_temp = []
        for j in range(len(mat1[i])):
            new_temp.append(mat1[i][j])
        new_mat.append(new_temp)
    if axis == 1:
        for i in range(len(mat2)):
            for j in range(len(mat2[i])):
                new_mat[i].append(mat2[i][j])
    elif axis == 0:
        for i in range(len(mat2)):
            new_temp = []
            for j in range(len(mat2[i])):
                new_temp.append(mat2[i][j])
        new_mat.append(new_temp)
    return new_mat
