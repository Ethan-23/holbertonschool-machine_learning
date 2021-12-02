#!/usr/bin/env python3
"""2-size_me_please"""


def matrix_shape(matrix):
    """Finds the shape of the matrix"""
    col = 0
    row = 0
    lenn = 0
    intlist = []
    for i in matrix:
        col += 1
    intlist.append(col)
    for j in i:
        row += 1
    intlist.append(row)
    if type(j) is list:
        for k in j:
            lenn += 1
        intlist.append(lenn)
        return intlist
    return intlist
