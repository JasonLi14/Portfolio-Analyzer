import yfinance as yf
import numpy as np
import numpy_financial  as npf
import pandas as pd
from Tools.scripts.dutree import display


# ticker_prices consumes a list of stocks, start date, end date, and an interval
# and produces a dataframe of prices for each of the stock
def ticker_prices(ticker_list, start, end, interval):
    prices = pd.DataFrame()

    hist_ticker = yf.Ticker(ticker_list[0])
    prices[ticker_list[0]] = hist_ticker.history(start=start, end=end, interval=interval).Close

    ticker_list.pop(0)

    for i in ticker_list:
        hist_ticker = yf.Ticker(i)
        prices[i] = hist_ticker.history(start=start, end=end, interval=interval).Close

    return prices


# sort_by_sharpe consumes a dataframe of stocks from the same industry and their monthly prices
# and produces a sorted dataframe by their sharpe ratio.
def sort_by_sharpe(stocks, min_sharpe, min_return, min_std):

    sharpe_df = pd.DataFrame(columns=['Returns', 'Std', 'Sharpe'])
    returns = 0
    std = 0

    stock_info = {}#fixed length list of Ticker, Returns, Std, Sharpe

    for ticker in stocks.columns:

        #work out equations
        returns = stocks.loc[stocks.index[-1], ticker] - stocks.loc[stocks.index[0], ticker]

        #work out equations
        std = stocks[ticker].std()
        sharpe = returns/std

        #print(ticker, sharpe, returns, std)

        if sharpe > min_sharpe and returns > min_return and std > min_std:
            stock_info = {
                'Returns': returns,
                'Std': std,
                'Sharpe': sharpe
            }

            # stock_info = pd.DataFrame(stock_info)

            sharpe_df.loc[ticker] = stock_info



    sharpe_df = sharpe_df.sort_values('Sharpe', ascending = False)

    return sharpe_df

def main():
    ticker_list = ['AAPL', 'SHOP.TO', 'TD.TO', 'MSFT', 'NVDA', 'NFE', 'FIVE', ]
    start_date = '2019-07-01'
    end_date = '2022-12-04'

    stock_prices = ticker_prices(ticker_list, start_date, end_date, '1mo')

    stock_sharpe = sort_by_sharpe(stock_prices, 1.5, 0, 0)

    print(stock_sharpe.to_string())


main()
