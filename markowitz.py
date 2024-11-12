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

from typing import Tuple

# other modules
import jason
import patrick


def getPortfolioResults(df: pd.DataFrame) -> Tuple:
    # Returns the average returns and standard deviation of the portfolio
    pf_value = df.sum(axis=1)
    pct_df = jason.convertToPct(pf_value)
    avg_returns = pct_df.mean()
    std = pct_df.std()
    return avg_returns, std


def applyWeightings(df: pd.DataFrame, weightings: list, investment: int) -> pd.DataFrame:
    # Returns the dataframe adjusted for all the weightings
    # Requires that df has the same number of rows as the length of weightings
    i = 0
    for column in df.columns.values:
        # find shares
        shares = investment / df[column].iloc[0]
        df[column] *= shares * weightings[i]
        i = i + 1
    return df


def getClosePrices(start: str, end: str, tickers: list, cutoff: str) -> pd.DataFrame:
    stock_data = yf.download(" ".join(tickers), start=start, end=end, interval='1mo')["Close"]
    stock_data.index = stock_data.index.strftime("%Y-%m-%d")
    # loop through tickers
    for column in stock_data.columns.values:
        # if the first close price is less than cutoff
        if np.isnan(stock_data.at[cutoff, column]):
            stock_data.drop(columns=[column], inplace=True)
    return stock_data


def simulateRandom(tests: int, stock_data: pd.DataFrame) -> Tuple[list, list]:
    # Simulates tests amount of tests with random weightings
    stocks_amount = len(stock_data.columns)
    results = [[], [], []]
    weighting_record = []
    for test in range(tests):  # simulate a set amount of tests
        weighting = np.random.random(size=stocks_amount)  # Find random weightings
        weighting /= np.sum(weighting)  # make sure it adds up to 1
        weighting = weighting.tolist()
        weighted_df = applyWeightings(stock_data, weighting, 1000000)  # Find df with weightings
        avg_return, std = getPortfolioResults(weighted_df)  # Find metrics for performance
        results[0].append(avg_return)
        results[1].append(std)
        results[2].append(avg_return / std * (12 ** 0.5))  # annualized Sharpe Ratio
        weighting_record.append(weighting)
    return results, weighting_record


def plotSimulation(results: list):
    x = results[1]
    y = results[0]
    colors = results[2]
    plt.scatter(x, y, c=colors, cmap='summer')
    plt.title("Graph of Various Weightings with the Portfolio")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Average Monthly Returns")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


if __name__ == "__main__":
    valid_tickers = ['LLY', 'ABBV', 'AAPL', 'BMY', 'UNH', 'UPS', 'CAT', 'TXN', 'PEP',
                     'RY.TO', 'ACN', 'PG', 'QCOM', 'MRK', 'T.TO', 'PM', 'BLK', 'TD.TO'
                     ]
    data = getClosePrices('2012-11-09', '2024-11-09', valid_tickers[:5], '2014-01-01')
    print(data.head())
    simulation_results, simulation_weights = simulateRandom(10000, data)
    plotSimulation(simulation_results)
