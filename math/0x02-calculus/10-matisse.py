#!/usr/bin/env python3
"""Task 10"""


def poly_derivative(poly):
    """poly derivative"""
    if type(poly) != list or len(poly) == 0:
        return None
    if poly == [0]:
        return [0]
    new = []
    for i in range(len(poly)):
        if i != 0:
            if poly[i] == 0:
                new.append(poly[i])
            else:
                new.append(poly[i] * i)
    return new
