from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import GetAssetsRequest
import pandas as pd
import numpy as np

TIMEFRAME_15_MIN = TimeFrame(15, TimeFrameUnit.Minute)
TIMEFRAME_1_DAY = TimeFrame(1, TimeFrameUnit.Day)


def hist_stock_data(client ,symbol, timeframe=TIMEFRAME_15_MIN, limit=None, start=None, end=None, after=None, until=None):
    '''
    Get historical data for a list of symbols
    :param client: Alpaca client
    :param symbols: List of symbols
    :param timeframe: Timeframe for the data
    :param limit: Limit of bars to return
    :param start: Start date for the data
    :param end: End date for the data
    :param after: Start date for the data
    :param until: End date for the data
    :return: Dictionary with the historical data for each symbol
    '''
    request_params = StockBarsRequest(
                        symbol_or_symbols= symbol,
                        timeframe=timeframe,
                        limit=limit,
                        start=start,
                        end=end,
                        after=after,
                        until=until
                    )

    bars = client.get_stock_bars(request_params)
    df = bars.df.reset_index()
    if len(df) == 0:
        return pd.DataFrame([0])
    print(df.head())
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_convert("America/Santiago")

    #df.between_time('09:31', '16:00')

    return df.sort_index(ascending=True)

def download_data(stock_to_download ,stock_client_for_download , folderPath='E:\stockdata', startdate='2016-06-01',enddate='2024-03-01'):
    result = hist_stock_data(stock_client_for_download, stock_to_download,start=startdate, end=enddate)
    result.to_csv(f'{folderPath}/{stock_to_download}.csv')
    del result
    return f"Data Downloaded Successfully {stock_to_download} between {startdate} and {enddate}"