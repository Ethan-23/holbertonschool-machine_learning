#!/usr/bin/env python3
"""task 17"""


def poly_integral(poly, C=0):
    """Integrate"""
    if type(poly) != list or len(poly) == 0 or type(C) is not int:
        return None
    if len(poly) == 1:
        return [C]
    integral = [C]
    for i in range(len(poly)):
        num = poly[i] / (i + 1)
        if num.is_integer() is False:
            integral.append(num)
        else:
            integral.append(int(num))
    return integral
