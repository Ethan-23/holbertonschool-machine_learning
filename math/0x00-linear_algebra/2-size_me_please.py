#!/usr/bin/env python3
"""2-size_me_please"""

def matrix_shape(matrix):
    """Finds the shape of the matrix"""
    col = 0
    row = 0
    lenn = 0
    for i in matrix:
        col += 1
    for j in i:
        row += 1
    if type(j) is list:
        for k in j:
            lenn += 1
        return col, row, lenn
    return col, row
