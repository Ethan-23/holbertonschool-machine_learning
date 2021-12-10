#!/usr/bin/env python3
"""task 17"""


def poly_integral(poly, C=0):
    """Integrate"""
    if type(poly) != list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    integral = [0]
    for i in range(len(poly)):
        if poly[i] == 0 or i == 0:
            integral.append(int(poly[i]))
        else:
            integral.append(poly[i] / (i + 1))
    return integral
