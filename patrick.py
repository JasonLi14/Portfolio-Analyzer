import yfinance as yf
import numpy as np
import numpy_financial  as npf
import pandas as pd
from Tools.scripts.dutree import display
import jason


def ticker_prices(ticker_list, start, end, interval):
    """
        This function, when given a list of tickers, a start date, end date, and interval
        will return a dataframe of the prices of the tickers from the start date to the end
        date at the set interval

        :param ticker_list: int
        :param start: datetime
        :param end: datetime
        :return: dataframe
        """

    prices = pd.DataFrame()

    hist_ticker = yf.Ticker(ticker_list[0])
    prices[ticker_list[0]] = hist_ticker.history(start=start, end=end, interval=interval).Close

    ticker_list.pop(0)

    for i in ticker_list:
        hist_ticker = yf.Ticker(i)
        prices[i] = hist_ticker.history(start=start, end=end, interval=interval).Close

    return prices

#feed pct_change() data
def sort_by_sharpe(price_pct, min_sharpe, min_return, max_std):
    """
        This function, when given a dataframe of
        price percent change, a dataframe of prices, a minimum sharpe ratio, a minimum return
        and maximum standard deviation will produce a dataframe with the returns, standard deviation
        sharpe ratio and prices of all stocks in the given dataframe that meets the set criteria

        :param price_pct: dataframe
        :param min_sharpe: float
        :param min_return: float
        :param: max_std: float
        :return: dataframe
        """

    sharpe_df = pd.DataFrame(columns=['Returns', 'Std', 'Sharpe'])
    returns = 0
    std = 0
    ticker = ""

    stock_info = {}#fixed length list of Ticker, Returns, Std, Sharpe

    tick_list = price_pct.columns

    for i in range(len(tick_list)):
        ticker = tick_list[i]

        #work out equations - pct_change() on all,
        returns = price_pct[ticker].mean()

        #work out equations
        std = price_pct[ticker].std()

        #multiply sharpe_ratio by sqrt(15)
        sharpe = returns/std * (50 ** 0.5)

        #print(ticker, sharpe, returns, std)

        if sharpe > min_sharpe and returns > min_return and std < max_std:
            stock_info = {
                'Returns': returns,
                'Std': std,
                'Sharpe': sharpe
            }

            # stock_info = pd.DataFrame(stock_info)

            sharpe_df.loc[ticker] = stock_info
        elif tick_list.size - sharpe_df.index.size < 12:
            stock_info = {
                'Returns': returns,
                'Std': std,
                'Sharpe': sharpe
            }

            # stock_info = pd.DataFrame(stock_info)

            sharpe_df.loc[ticker] = stock_info

    sharpe_df = sharpe_df.sort_values('Sharpe', ascending = False)

    return sharpe_df

def keep_tickers(dataframe, list):
    """
    This function, given a dataframe and list of tickers, will keep
    all items in the dataframe with a ticker in the list. Tickers must be in
    the Dataframe

    :param dataframe: pd.DataFrame
    :param list: list[Str]
    :return: pd.DataFrame
    """

    newframe = pd.DataFrame()

    for i in list:
        newframe[i] = dataframe[i]

    return newframe

def stock_df_to_ticker(dataframe):
    """
        This function, when given a dataframe of stocks, will return a list of ticker strings

        :param dataframe: dataframe
        :return: list[str]
        """

    ticker_list = []

    for i in dataframe.index:
        ticker_list.append(i)

    return ticker_list

def filtering(list_len, stock_correlation_tiers):
    """
    This function, when given how long the list will be and categorization of stocks,
    will return a list of stocks that we want to craft the portfolio from.
    :param list_len: int
    :param stock_correlation_tiers: list[list[str]]
    :return: list[str]
    """
    ticker_list = []
    '''
    while 0 < len(stock_correlation_tiers) and len(ticker_list) < list_len:
        sub_list = stock_correlation_tiers[0]
        while 0 < len(sub_list) and len(ticker_list) < list_len:
            print(sub_list[0])
            ticker_list.append(sub_list[0])
            sub_list.pop(0)

        stock_correlation_tiers.pop(0)
    '''

    while len(ticker_list) < list_len:
        for i in range(len(stock_correlation_tiers)):
            sub_list = stock_correlation_tiers[i]

            if len(sub_list) > 0:
                ticker_list.append(sub_list[0])
                sub_list.pop(0)
                stock_correlation_tiers[i] = sub_list

    return ticker_list

def arrange_by_sharpe(prices:pd.DataFrame, sharpe:pd.DataFrame):
    """
    This function, when given how long the list will be and categorization of stocks,
    will return a list of stocks that we want to craft the portfolio from.
    :param prices: pd.DataFrame
    :param sharpe: pd.DataFrame
    :return: pd.DataFrame
    """
    new_prices = pd.DataFrame()

    for i in sharpe.index:
        #print(i)
        #print(prices[i])
        new_prices[i] = prices[i]

    #print(new_prices)

    return new_prices

def correlation_filter(prices: pd.DataFrame, max_corr: float):
    """
    This function, when given a dataframe of prices and

    :param prices: pd.DataFrame
    :param max_corr: float
    :return: list[Str]
    """
    correlations = prices.corr()

    tickers = []

    corr_list = correlations.index

    for i in range(len(corr_list)):
        tick = corr_list[i]

        if correlations.loc[tick].mean() <= max_corr:
            tickers.append(tick)
        elif corr_list.size - len(tickers) < 12:
            tickers.append(tick)

    return tickers

def main():
    '''ticker_list = ['AAPL', 'SHOP.TO', 'TD.TO', 'MSFT', 'NVDA', 'NFE', 'FIVE', 'GOEV', 'LI', 'APA', 'ACHC', 'IMAB',
                   'REAL', 'BYND', ]'''

    #GATEEK: produces a dataframe with the price of a stock on a date range

    file_tickers = jason.getAllTickers("Tickers_Example.csv").tolist()

    start_date = '2020-07-01'
    end_date = '2024-11-10'

    stock_prices = ticker_prices(file_tickers, start_date, end_date, '1wk')

    stock_pct_change = stock_prices.pct_change()

    stock_pct_change = keep_tickers(stock_pct_change, correlation_filter(stock_pct_change, 0.75))

    stock_sharpe = sort_by_sharpe(stock_pct_change, 0.25, 0, 1)

    #print(stock_sharpe)

    best_stock = stock_sharpe.index[0]

    stock_prices = arrange_by_sharpe(stock_prices, stock_sharpe)

    stock_correlation_tiers = jason.categorize(stock_prices, best_stock, 10)

    #print(stock_sharpe)
    #print(stock_correlation_tiers)

    ticker_lst = [best_stock]

    ticker_lst += filtering(24, stock_correlation_tiers)

    print(ticker_lst)
