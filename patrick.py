import yfinance as yf
import numpy as np
import numpy_financial as npf
import pandas as pd



#short by sharpe consumes a dataframe of stocks from the same industry and their daily prices prices
#and produces a sorted dataframe by their sharpe ratio.
def sort_by_sharpe(dataframe):
  
