import pandas as pd
import yfinance as yf
import numpy as np
from tabulate import tabulate


def display(df):
    """
    Uses tablature to output a dataframe nicely.
    :param df: pd.DataFrame
    :return: None
    """
    print(tabulate(df, headers='keys', tablefmt='psql'))


def getStockData(ticker, start_date="2014-01-01", end_date="2024-11-05", interval="1mo"):
    """
    This will get data for a ticker and return it in a nice dataframe.
    :param ticker: string
    :param start_date: string, must be "YYYY-MM-DD"
    :param end_date: string, must be "YYYY-MM-DD"
    :param interval: string
    :return: pd.DataFrame
    """
    close_history = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)["Close"]
    # Make the date look nice
    close_history.index = close_history.index.strftime("%Y-%m-%d")
    df = pd.DataFrame(close_history)
    df.columns = [ticker]
    return df


def convertToPct(df):
    """
    Converts a stock dataframe to percentage changes
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    pct_df = df.pct_change() * 100
    # Drop first row
    pct_df.drop(index=pct_df.index[0], inplace=True)
    return pct_df


def getPctStats(df):
    """
    Gets some statistics that are percentage based.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    avg_returns = df.mean()
    std = df.std()
    sharpe = avg_returns/std
    # Formatting
    indices = df.columns.values
    data = {"Average Returns": avg_returns,
            "Standard Deviation": std,
            "Sharpe Ratio": sharpe}
    return pd.DataFrame(data, index=indices)


if __name__ == "__main__":
    test_stock = getStockData("AAPL")
    display(test_stock.head())
    test_stock_pct = convertToPct(test_stock)
    display(test_stock_pct.head())
    apple_pct_stats = getPctStats(test_stock_pct)
    display(apple_pct_stats)
