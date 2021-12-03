#!/usr/bin/env python3
"""7-getting_cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """cat_matrices2D"""
    new_mat = []
    for i in mat1:
        temp_mat = i.copy()
        new_mat.append(temp_mat)
    if axis == 0:
        for i in mat2:
            temp_mat = i.copy()
            new_mat.append(temp_mat)
        return new_mat
    else:
        for i in range(len(mat2)):
            for j in range(len(mat2[i])):
                new_mat[i].append(mat2[i][j])
        return new_mat
