#!/usr/bin/env python3


def poly_derivative(poly):
    """poly derivative"""
    new = []
    for i in range(len(poly)):
        if i != 0:
            if poly[i] == 0:
                new.append(poly[i])
            else:
                new.append(poly[i] * i)
    return new
