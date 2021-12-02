#!/usr/bin/env python3
"""8-ridin_bareback"""


def mat_mul(mat1, mat2):
    """mat_mul"""
    new_mat = []
    for col in range(len(mat1)):
        new_temp = []
        for lenn in range(len(mat2[0])):
            number = 0
            for row in range(len(mat1[col])):
                number += mat1[col][row] * mat2[row][lenn]
            new_temp.append(number)
        new_mat.append(new_temp)
    return new_mat
