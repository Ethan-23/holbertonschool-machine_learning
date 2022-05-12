#!/usr/bin/env python3
"""Processing the data"""

import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime


def preprocess_data():
    """Cleans up raw data"""
    filename = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip"
    path = os.getcwd() + "/data/" + filename
    csv_path, _ = os.path.splitext(path)
    df = pd.read_csv(csv_path)

    df = df.dropna()
    length = len(df)
    train_df = df[0:int(length*0.7)]
    val_df = df[int(length*0.7):int(length*0.9)]
    test_df = df[int(length*0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
