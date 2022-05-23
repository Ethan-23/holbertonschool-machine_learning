#!/usr/bin/env python3
"""Positional Encoding"""

import numpy as np


def calc(i, j, dm):
    return i * (1 / (10000) ** (j / dm))


def positional_encoding(max_seq_len, dm):
    """positional_encoding"""
    encoding = np.zeros([max_seq_len, dm])

    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            encoding[i, j] = np.sin(calc(i, j, dm))
            encoding[i, j + 1] = np.cos(calc(i, j, dm))
    return encoding
