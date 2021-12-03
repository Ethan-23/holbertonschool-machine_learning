#!/usr/bin/env python3
"""2-size_me_please"""


def matrix_shape(matrix):
    """Finds the shape of the matrix"""
    intlist = []
    intlist.append(len(matrix))
    if type(matrix[0]) is list:
        intlist.append(len(matrix[0]))
        if type(matrix[0][0]) is list:
            intlist.append(len(matrix[0][0]))
    return intlist
