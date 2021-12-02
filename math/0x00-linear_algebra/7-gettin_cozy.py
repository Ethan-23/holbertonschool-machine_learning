#!/usr/bin/env python3
"""7-getting_cozy"""

def cat_matrices2D(mat1, mat2, axis=0):
    """cat_matrices2D"""
    new_mat = []
    for i in mat1:
        new_temp = []
        for j in i:
            new_temp.append(j)
        new_mat.append(new_temp)
    if axis == 1:
        for i in range(len(mat2)):
            for j in mat2[i]:
                new_mat[i].append(j)
    elif axis == 0:
        for i in mat2:
            new_temp = []
            for j in i:
                new_temp.append(j)
        new_mat.append(new_temp)
    return new_mat
