#!/usr/bin/env python3
"""12-bracin_the_elements"""

def np_elementwise(mat1, mat2):
    """np_elementwise"""
    add = mat1 + mat2
    subtract = mat1 - mat2
    divide = mat1 / mat2
    mul = mat1 * mat2
    return(add, subtract, mul, divide)
