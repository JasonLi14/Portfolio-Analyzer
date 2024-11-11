import yfinance as yf
import numpy as np
import numpy_financial  as npf
import pandas as pd
from Tools.scripts.dutree import display
import jason


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
def sort_by_sharpe(stocks, min_sharpe, min_return, max_std):

    sharpe_df = pd.DataFrame(columns=['Returns', 'Std', 'Sharpe'])
    returns = 0
    std = 0

    stock_info = {}#fixed length list of Ticker, Returns, Std, Sharpe

    for ticker in stocks.columns:

        #work out equations - pct_change() on all,
        returns = stocks[ticker].pct_change(fill_method=None).mean()

        #work out equations
        std = stocks[ticker].pct_change(fill_method=None).std()

        #multiply sharpe_ratio by sqrt(15)
        sharpe = returns/std * 52

        #print(ticker, sharpe, returns, std)

        if sharpe > min_sharpe and returns > min_return and std < max_std:
            stock_info = {
                'Returns': returns,
                'Std': std,
                'Sharpe': sharpe
            }

            # stock_info = pd.DataFrame(stock_info)

            sharpe_df.loc[ticker] = stock_info



    sharpe_df = sharpe_df.sort_values('Sharpe', ascending = False)

    return sharpe_df

#takes a dataframe and converts it to a list of tickers.
def stock_df_to_ticker(dataframe):
    ticker_list = []

    for i in dataframe.index:
        ticker_list.append(i)

    return ticker_list

'''
def main():

    ticker_list = ['AAPL', 'SHOP.TO', 'TD.TO', 'MSFT', 'NVDA', 'NFE', 'FIVE', 'GOEV', 'LI', 'APA', 'ACHC', 'IMAB',
                   'REAL', 'BYND']

    #ticker_list = jason.getAllTickers("Tickers_Example.csv").tolist()

    #print(ticker_list)


    start_date = '2019-07-01'
    end_date = '2022-12-04'

    stock_prices = ticker_prices(ticker_list, start_date, end_date, '1wk')

    #print(stock_prices)


    stock_sharpe = sort_by_sharpe(stock_prices, 0, 0, 1)

    ticker_list = stock_df_to_ticker(stock_sharpe)

    #print(stock_sharpe)
'''

def filtering(list_len, stock_correlation_tiers):
    ticker_list = []

    i = 0

    while i < len(stock_correlation_tiers) and len(ticker_list) < list_len:
        ticker_list += stock_correlation_tiers[0]
        #print(type(stock_correlation_tiers))

        stock_correlation_tiers.pop(0)

        i += 1

    return ticker_list

def main():
    '''ticker_list = ['AAPL', 'SHOP.TO', 'TD.TO', 'MSFT', 'NVDA', 'NFE', 'FIVE', 'GOEV', 'LI', 'APA', 'ACHC', 'IMAB',
                   'REAL', 'BYND', ]'''

    ticker_list = jason.getAllTickers("Tickers_Example.csv").tolist()

    #print(ticker_list)

    start_date = '2019-07-01'
    end_date = '2022-12-04'

    stock_prices = ticker_prices(ticker_list, start_date, end_date, '1wk')

    # print(stock_prices)

    #might
    stock_sharpe = sort_by_sharpe(stock_prices, 0, 0, 1)

    best_stock = stock_sharpe.index[0]

    print(best_stock)


    ticker_lst = stock_df_to_ticker(stock_sharpe)
    print(ticker_lst)

    test_stock = jason.getStockData(ticker_lst[1])

    test_stock_pct = jason.convertToPct(test_stock)

    test_stock_pct_stats = jason.getPctStats(test_stock_pct)

    #multi_stock_df = pd.concat([jason.getStockData(ticker_lst[i]) for i in [0, 1, 2, 3, 5, 6, 7, 8, 9]], axis=1)
    multi_stock_pct = jason.convertToPct(stock_prices)

    stock_correlation_tiers = jason.categorize(multi_stock_pct, best_stock, 10)

    print(stock_correlation_tiers)

    ticker_lst = [best_stock]

    ticker_lst += filtering(15, stock_correlation_tiers)


    print(ticker_lst)

main()
