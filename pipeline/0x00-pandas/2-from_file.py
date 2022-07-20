#!/usr/bin/env python3
"""From Numpy"""

import pandas as pd


def from_file(filename, delimiter):
    """from_numpy"""
    return pd.read_csv(filename, delimiter=None)
