import yfinance as yf
import numpy as np
import numpy_financial as npf
import pandas as pd



#ticker_prices consumes a list of stocks, start date, end date, and an interval
# and produces a dataframe of prices for each of the stock
def ticker_prices(ticker_list, start, end, interval):
    

#sort_by_sharpe consumes a dataframe of stocks from the same industry and their daily prices prices
#and produces a sorted dataframe by their sharpe ratio.
def sort_by_sharpe(dataframe):
  
def main():
  ticker_list = ['AAPL', 'SHOP.TO', 'TD.TO', 'MSFT', 'NVDA']
  start_date = '2019-07-01'
  end_date = '2024-11-04'

