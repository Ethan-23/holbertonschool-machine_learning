#!/usr/bin/env python3
"""Something"""


def summation_i_squared(n):
    """summation_i_squared"""
    if type(n) != int or n < 1:
        return None
    else:
        return int((n * (n+1) * (2*n+1)) / 6)
