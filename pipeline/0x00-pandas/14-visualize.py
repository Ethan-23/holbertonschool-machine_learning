#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])

df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')
df['Date'] = df['Date'].dt.to_period('d')

df = df.loc[df['Date'] >= "2017-01-01"]
df = df.set_index('Date')

df['Close'].fillna(method='pad', inplace=True)

df.fillna(
    {
        'High': df.Close,
        'Low': df.Close,
        'Open': df.Close,
        'Volume_(BTC)': 0,
        'Volume_(Currency)': 0
    },
    inplace=True
)

df_plot = pd.DataFrame()
df_plot['High'] = df['High'].resample('d').max()
df_plot['Low'] = df['Low'].resample('d').min()
df_plot['Open'] = df['Open'].resample('d').mean()
df_plot['Close'] = df['Close'].resample('d').mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('d').sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('d').sum()

df_plot.plot()
plt.show()
