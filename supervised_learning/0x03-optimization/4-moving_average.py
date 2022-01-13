#!/usr/bin/env python3
"""4-moving_average"""
import numpy as np


def moving_average(data, beta):
    """
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Returns: a list containing the moving averages of data
    """
    v = 0
    AVG = []
    for i in range(len(data)):
        v = (v * beta) + ((1 - beta) * data[i])
        AVG.append(v / (1 - (beta ** (i + 1))))
    return AVG
