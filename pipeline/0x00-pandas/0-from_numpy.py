#!/usr/bin/env python3
"""From Numpy"""

import pandas as pd


def from_numpy(array):
    """from_numpy"""
    df = pd.DataFrame(array)
    df = df.rename(columns={i:chr(i+65) for i in range(26)})
    return df
