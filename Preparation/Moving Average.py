import yfinance as yf
import numpy as np
import numpy_financial  as npf
import pandas as pd
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

#last month vs. last week
def moving_avg(prices, long_interval, short_interval):
    """
    This function, when given a dataframe of prices, and interval
    will return a dataframe of the moving average of the prices covering the listed interval

    :param prices: dataframe
    :param long_interval: int
    :param short_interval: int
    :return: dataframe
    """

    averaged_prices = pd.DataFrame(columns = ['long', 'short', 'difference'])

    for stock in prices.columns:
        sum_long = 0
        num_days = prices.index.size

        for i in range(long_interval):
            sum_long += prices.loc[prices.index[num_days-i-1], stock]

        sum_long = sum_long / long_interval

        averaged_prices.loc[stock, 'long'] = sum_long

        sum_short = 0
        num_days = prices.index.size

        for i in range(short_interval):
            sum_short += prices.loc[prices.index[num_days - i-1], stock]

        sum_short = sum_short / short_interval

        averaged_prices.loc[stock, 'short'] = sum_short

        averaged_prices.loc[stock, 'difference'] = (averaged_prices.loc[stock, 'short'] - averaged_prices.loc[stock, 'long'])/ prices.loc[prices.index[num_days-1], stock]

    averaged_prices = averaged_prices.sort_values('difference', ascending=False)

    return averaged_prices

file_tickers = jason.getAllTickers("Tickers_Example.csv").tolist()

start_date = '2020-07-01'
end_date = '2024-11-10'

test_stock = ticker_prices(file_tickers, start_date, end_date, '1wk')

stock_prices = moving_avg(test_stock, 30, 7)

print(stock_prices.to_string())