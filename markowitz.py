"""
Title: Markowitz Implementation
Author: Jason
Date Created: 2024-11-11
"""

# Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# other modules
import jason
import patrick


def getClosePrices(start, end, tickers, cutoff):
    stock_data = yf.download(" ".join(tickers), start=start, end=end, interval='1mo')["Close"]
    stock_data.index = stock_data.index.strftime("%Y-%m-%d")

    print(stock_data.index)
    # loop through tickers
    for column in stock_data.columns.values:
        # if the first close price is less than cutoff
        if np.isnan(stock_data.at[cutoff, column]):
            stock_data.drop(columns=[column], inplace=True)

    return stock_data


if __name__ == "__main__":
    valid_tickers = ['LLY', 'ABBV', 'AAPL', 'BMY', 'UNH', 'UPS', 'CAT', 'TXN', 'PEP',
                     'RY.TO', 'ACN', 'PG', 'QCOM', 'MRK', 'T.TO', 'PM', 'BLK', 'TD.TO'
                     ]
    data = getClosePrices('2014-11-09', '2024-11-09', valid_tickers, '2015-01-01')
    jason.display(data)
    plt.plot(data["RY.TO"])
    plt.show()
