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


# FROM GATEEK's IPYNB
def valid_stocks(tickers_file):
    # Read CSV and get tickers
    tickers_df = pd.read_csv(tickers_file)
    tickers_df.columns = (['Tickers'])
    tickers_list = tickers_df['Tickers'].tolist()

    # Start and end dates
    start = '2023-10-01'
    end = '2024-09-30'

    valid_tickers = []

    for ticker in tickers_list:
        # Loads in ticker info from yfinance
        stock = yf.Ticker(ticker)
        info = stock.fast_info 

        # filter ticker by currency
        try:
            currency = info['currency']
        except:
            continue

        if currency != 'USD' and currency != 'CAD':
            continue

        #filter ticker by average monthly volume
        try:
            hist = stock.history(start=start, end=end, interval='1d')
        except:
            continue
        monthly_volume = pd.DataFrame()
        monthly_volume['volume'] = hist['Volume'].resample('M').sum()
        monthly_volume['count'] = hist['Volume'].resample('M').count()
        monthly_volume['avg monthly volume'] = monthly_volume['volume'] / monthly_volume['count']
        invalid_trading_days = monthly_volume[monthly_volume['count'] < 18]
        invalid_monthly_vol = monthly_volume[monthly_volume['avg monthly volume'] < 100000]

        if len(invalid_monthly_vol) > 0 or len(invalid_trading_days) > 0:
            continue


        valid_tickers.append(ticker)

    return valid_tickers


def get_close_prices(start, end, tickers, cutoff):

    multi_data = pd.DataFrame()
    df = []
    appended_tickers = []

    # loop through tickers 
    for ticker in tickers:
        # get all data and put into a series
        data = yf.download(ticker, start=start, end=end, interval='1d')
        close = data['Close']
        close = close.rename(ticker)

        # if the first close price is less than cutoff
        if close.index.min() < pd.Timestamp(cutoff):
            # add stock close prices to df
            df.append(close)
            appended_tickers.append(ticker)

    # create df with all the data
    multi_data = pd.concat(df, axis=1)
    #drop all values so that there are valid data points for each date in the index
    multi_data.dropna(subset=appended_tickers, inplace=True)

    # Get CAD->USD exchange rate
    cadusd = yf.download('CAD=x', start=start, end=end, interval='1d')

    # convert everything to CAD
    for ticker in appended_tickers:
        stock = yf.Ticker(ticker)
        info = stock.fast_info

        if info['currency'] == 'USD':
            multi_data[ticker] = multi_data[ticker] * cadusd['Close']
    
    return multi_data


def buy_shares(weightings_df, prices_df):

    cash = 1000000
    flat_fee = 3.95
    fee_per_share = 0.001

    weightings_df['Close Price'] = prices_df.reindex(weightings_df.index)

    # 1: Calculate the initial investment of each stock and the amount of shares
    weightings_df['Investment Amt'] = cash * (weightings_df['Weight'] / 100)
    weightings_df['Shares'] = weightings_df['Investment Amt'] / weightings_df['Close Price']

    # 2: Calculate the fees based on what kind of fee structure is cheaper
    weightings_df['fees'] = np.minimum(weightings_df['Shares'] * fee_per_share, flat_fee)

    # 3: Calculate total investment with fees added
    weightings_df['Investment with fees'] = weightings_df['Shares'] * weightings_df['Close Price'] + weightings_df['fees']
    total_with_fees = weightings_df['Investment with fees'].sum()

    # 4: Adjust investment to keep the total under the budget
    adjustment_factor = cash / total_with_fees
    weightings_df['Adjusted Investment Amt'] = weightings_df['Investment Amt'] * adjustment_factor
    weightings_df['Adjusted Shares'] = weightings_df['Adjusted Investment Amt'] / weightings_df['Close Price']

    # 5: Recalculate fees
    weightings_df['Adjusted fees'] = np.minimum(weightings_df['Adjusted Shares'] * fee_per_share, flat_fee)

    # 6: Final investment for each stock
    weightings_df['Final Investment'] = weightings_df['Adjusted Shares'] * weightings_df['Close Price'] + weightings_df['Adjusted fees']

    # Create Final Portfolio
    Portfolio_Final = pd.DataFrame()
    Portfolio_Final['Ticker'] = weightings_df.index
    Portfolio_Final.index = Portfolio_Final['Ticker']
    Portfolio_Final['Price'] = weightings_df['Close Price']
    Portfolio_Final['Currency'] = 'CAD' # NEED TO FIGURE OUT A WAY TO GET ACCURATE CURRENCY DATA
    Portfolio_Final['Shares'] = weightings_df['Adjusted Shares']
    Portfolio_Final['Value'] = weightings_df['Adjusted Investment Amt']
    Portfolio_Final['Weight'] = weightings_df['Weight']

    Portfolio_Final.index = range(1, len(Portfolio_Final) + 1)

    return Portfolio_Final


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
