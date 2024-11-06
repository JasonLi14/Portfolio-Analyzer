import pandas as pd
import yfinance as yf
import numpy as np
from tabulate import tabulate


def getAllTickers(fileName="Tickers_Example.csv"):
    """
    Gets all the tickers from the csv file
    :param fileName: str
    :return: list[str]
    """
    return pd.read_csv(fileName, header=None)[0].values


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


def getStats(df: pd.DataFrame):
    """
    Gets some basic statistics that are value based.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    std = df.std()
    mean = df.mean()
    median = df.median()
    maximum = df.max()
    minimum = df.min()
    stats_df = pd.concat([std, mean, median, maximum, minimum], axis=1)
    stats_df.columns = ["Standard Deviation", "Mean", "Median", "Max", "Min"]
    return stats_df


def getPctStats(df):
    """
    Gets some statistics that are percentage based.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """

    avg_returns = df.mean()
    std = df.std()
    sharpe = avg_returns/std * (12 ** 0.5)
    # Formatting
    indices = df.columns.values
    data = {"Average Returns": avg_returns,
            "Standard Deviation": std,
            "Sharpe Ratio": sharpe}
    return pd.DataFrame(data, index=indices)


def getBeta(df, start_date="2014-01-01", end_date="2024-11-05", interval="1mo"):
    """
    # Get covariance with the market.
    :param df: pd.DataFrame
    :param start_date: string, must be "YYYY-MM-DD"
    :param end_date: string, must be "YYYY-MM-DD"
    :param interval: string
    :return: pd.DataFrame
    """
    market_data = getStockData("^GSPC", start_date=start_date, end_date=end_date, interval=interval)
    market_pct = convertToPct(market_data)
    compare_df = df.copy()
    compare_df["Market"] = market_pct
    stats_df = pd.DataFrame({"Covariance": compare_df.cov().loc[:, "Market"]})
    market_var = market_pct.std()["^GSPC"] ** 2
    stats_df["Beta"] = stats_df["Covariance"] / market_var
    return stats_df


def categorize(df:pd.DataFrame, pivot:str, categories:int):
    """
    Categorizes stocks into different categories depending on how correlated they are
    :param df: pd.DataFrame, must have pct data
    :param pivot: string that is in df
    :param categories: int
    :return: list[list[str]]
    """
    correlations = df.corr()
    lin_space = np.linspace(0, 0.9999999, categories + 1)[1:]
    # The 0.999999 prevents adding the stock itself into the list
    stock_categories = []

    for i in range(categories):
        stock_categories.append([])

    for stock in correlations.index:

        correlation = correlations.at[pivot, stock]
        inserted = False
        i = 0
        while i < categories and not inserted:
            if correlation <= lin_space[i]:
                stock_categories[i].append(stock)
                inserted = True
            i = i + 1
    return stock_categories


if __name__ == "__main__":
    # Just to test my functions
    ticker_lst = getAllTickers()
    test_stock = getStockData(ticker_lst[1])
    # display(test_stock.head())

    test_stock_pct = convertToPct(test_stock)
    # display(test_stock_pct.head())

    test_stock_pct_stats = getPctStats(test_stock_pct)
    # display(test_stock_pct_stats)

    # display(getBeta(test_stock_pct))
    # display(getStats(test_stock))

    multi_stock_df = pd.concat([getStockData(ticker_lst[i]) for i in [0, 1, 2, 3, 5, 6, 7, 8, 9]], axis=1)
    multi_stock_pct = convertToPct(multi_stock_df)
    print(categorize(multi_stock_pct, "AAPL", 10))
