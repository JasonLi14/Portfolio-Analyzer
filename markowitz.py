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
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from typing import Tuple, List

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


def getRandomWeightings(length: int, min_weight: float = 0) -> List[float]:
    # Returns a list of random weightings
    # Requires that the min_weight * len <= 1
    weightings_lst = np.random.random(size=length)  # Find random weightings
    weight_remainder = 1 - min_weight
    # Make sure weightings_lst sums up to weight remainder
    weightings_lst /= np.sum(weightings_lst)
    weightings_lst *= 1 - min_weight
    weightings_lst += min_weight
    return weightings_lst.tolist()


def simulateRandom(tests: int, stock_data: pd.DataFrame) -> Tuple[list, list, float, float]:
    # Simulates tests amount of tests with random weightings
    stocks_amount = len(stock_data.columns)
    results = [[], [], []]
    min_std = 1000  # We want to find the minimum and maximum standard deviations later
    max_std = -1000
    weighting_record = []
    for test in range(tests):  # simulate a set amount of tests
        weightings = getRandomWeightings(stocks_amount, 0.03)
        weighted_df = applyWeightings(stock_data, weightings, 1000000)  # Find df with weightings
        avg_return, std = getPortfolioResults(weighted_df)  # Find metrics for performance
        results[0].append(avg_return)
        results[1].append(std)
        results[2].append(avg_return / std * (12 ** 0.5))  # annualized Sharpe Ratio
        min_std = min(std, min_std)
        max_std = max(std, max_std)
        weighting_record.append(weightings)
    return results, weighting_record, min_std, max_std


def plotSimulation(results: list):
    x = results[1]
    y = results[0]
    colors = results[2]
    plt.scatter(x, y, c=colors, cmap='summer')
    plt.title("Graph of Various Weightings with the Portfolio")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Average Monthly Returns")
    plt.colorbar(label="Sharpe Ratio")


def findEfficientFrontier(results: list, weights: list, min_std: float, max_std: float):
    # This function will go through the results and pick the ones with the highest return
    # Per "group" of standard deviations.
    # Create groups
    result_groups = []
    start = round(min_std * 10)
    i = start  # Create groups that are 0.1 apart to make it easy to sort
    std_groups = [i/10]
    while i <= max_std * 10:
        i = i + 1
        std_groups.append(i/10)
        result_groups.append([])
    result_groups.append([])  # One more
    # Put results into groups

    amt_groups = len(result_groups) - 1
    for i in range(len(results[1])):
        result_i = round(results[1][i] * 10) - start
        # Insert a tuple with relevant information
        if result_i <= amt_groups:
            result_groups[result_i].append((results[0][i],
                                            results[1][i],
                                            results[2][i],
                                            weights[i]))

    # Find the most efficient in each category, i.e. highest return, and insert them into a list
    best_portfolios = []
    for group in result_groups:
        highest_return = 0
        best_portfolio = None
        for portfolio in group:
            if portfolio[0] > highest_return:  # The portfolio has higher returns
                highest_return = portfolio[0]
                best_portfolio = portfolio
        if best_portfolio is not None:  # Do not add from categories without a single portfolio
            best_portfolios.append(best_portfolio)
    return best_portfolios  # Notice that this is already sorted by risk


def optimizedEF(stocks: pd.DataFrame):
    # This will return the efficient frontier (i.e. most return for different amount of risk)
    # We are using the pypfopt library
    returns = expected_returns.mean_historical_return(stocks)
    S = risk_models.sample_cov(stocks)
    EF = EfficientFrontier(returns, S)
    return EF


def getRiskAdjustedPf(preference: float, best_portfolios: list) -> tuple:
    # Requires that scale is from 0 to 1
    # This will get the weighting of the portfolio for the risk we want
    pf_lst_len = len(best_portfolios) - 1  # We need to minus one to not get past the last index
    preferred_index = round(pf_lst_len * preference)
    return best_portfolios[preferred_index]


if __name__ == "__main__":
    valid_tickers = ['LLY', 'ABBV', 'AAPL', 'BMY', 'UNH', 'UPS', 'CAT', 'TXN', 'PEP',
                     'RY.TO', 'ACN', 'PG', 'QCOM', 'MRK', 'T.TO', 'PM', 'BLK', 'TD.TO'
                     ]
    data = getClosePrices('2012-11-09', '2024-11-09', valid_tickers[3:10], '2014-01-01')
    """
    simulation_results, simulation_weights, min_risk, max_risk = simulateRandom(10, data)

    plotSimulation(simulation_results)
    best_pfs = findEfficientFrontier(simulation_results, simulation_weights, min_risk, (max_risk + min_risk)/2)
    # Plot the best portfolios
    best_pfs_x = [pf[1] for pf in best_pfs]  # Get all the x values
    best_pfs_y = [pf[0] for pf in best_pfs]
    plt.scatter(best_pfs_x, best_pfs_y, label="Optimal Portfolios")
    plt.legend()
    plt.show()

    # Find the portfolio we want
    efficient_weightings = [pf[3] for pf in best_pfs]
    chosen_weighting = getRiskAdjustedPf(1, efficient_weightings)
    print(chosen_weighting)
    """
    efficientFrontier = optimizedEF(data)
    best_sharpe = efficientFrontier.max_sharpe()
    cleaned_weights = efficientFrontier.clean_weights()
    print(cleaned_weights)
