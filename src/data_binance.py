import ccxt
import time
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
import os

from config import *

binance_exchange = ccxt.binanceus({
    'timeout': 15000,
    'enableRateLimit': True
    # 'options': {'defaultType': 'future'}
})

def download_data(start_date, end_date, interval='1d', filepath='../data/raw/', filename=None, save=True):
    '''
    start: str, start date in format 'YYYY-MM-DD'
    end: str, end date in format 'YYYY-MM-DD'
    interval: str, interval of data to download
    filepath: str, path to save the data in
    '''
    if filename is None:
        filename = f'master_crypto_data_{start_date}_{end_date}_{interval}.parquet'
    FILEPATH = filepath + filename
    
    active_usdt_tickers = get_tickers()
    all_data = []
    
    for ticker in tqdm(active_usdt_tickers, desc="Fetching crypto data"):
        try:
            cleaned_ticker = ticker.replace('/', '')
            df = fetch_data(cleaned_ticker, start_date, end_date, interval)
            all_data.append(df)
            print(f"Data for {cleaned_ticker} fetched successfully.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
        break
    
    # Concatenate all the DataFrames into one master DataFrame
    master_data = pd.concat(all_data, ignore_index=True)
    
    # Save the master DataFrame to a Parquet file
    if save:
        master_data.to_parquet(FILEPATH, index=False)
        print(f"Master parquet file saved: {FILEPATH}")
    
    return master_data

def get_tickers():
    markets = binance_exchange.load_markets()
    all_tickers = list(markets.keys())

    usdt_tickers = [ticker for ticker in all_tickers if ticker.endswith('USDT')]

    active_usdt_tickers = [
        ticker for ticker in usdt_tickers if markets[ticker]['active']
    ]
    
    return active_usdt_tickers

def fetch_data(ticker_symbol, start_date, end_date, interval):
    # Convert dates to timestamps in milliseconds
    start_timestamp = int(((pd.to_datetime(start_date)).tz_localize('UTC')).timestamp() * 1000)
    end_timestamp = int(((pd.to_datetime(end_date)).tz_localize('UTC')).timestamp() * 1000)

    # Obtain raw data from Binance US
    full_data_list = obtain_full_spotdata(start_timestamp, end_timestamp, binance_exchange, ticker_symbol, interval=interval)
    print(full_data_list)
    
    # Create a DataFrame with the correct column names
    data = pd.DataFrame(full_data_list, columns=SPOT_COLUMNS)
    data['Open time'] = data['Open time'].apply(lambda x: transform_timestamp(int(x)))

    # Remove any duplicate entries based on 'Open time'
    data.drop_duplicates('Open time', keep='first', inplace=True)

    # Add a column to identify the ticker
    data['Ticker'] = ticker_symbol
    return data

def get_spot(exchange, symbol, interval = '1d',
                    startTime = None,
                    endTime = None,
                    limit = 1000):

    if (startTime == None and endTime == None):
        return exchange.publicGetKlines({'symbol': symbol,
                                        'interval': interval,
                                        'limit': limit})
    elif (startTime == None and endTime != None):
        return exchange.publicGetKlines({'symbol': symbol,
                                        'interval': interval,
                                        'endTime': endTime,
                                        'limit': limit})
    elif (startTime != None and endTime == None):
        return exchange.publicGetKlines({'symbol': symbol,
                                        'interval': interval,
                                        'startTime': startTime,
                                        'limit': limit})
    else:
        return exchange.publicGetKlines({'symbol': symbol,
                                        'interval': interval,
                                        'startTime': startTime,
                                        'endTime': endTime,
                                        'limit': limit})

def convert_to_seconds(time_input):
    number = int(time_input[:-1])
    unit = time_input[-1]

    if unit == 's':
        return number
    elif unit == 'm':
        return number * 60
    elif unit == 'h':
        return number * 3600
    elif unit == 'd':
        return number * 86400
    else:
        raise ValueError("Unsupported time unit")
    
def transform_timestamp(timestamp_integer):
    '''
    As data points involved milliseconds, we need to transform them by constant 1000.
    '''

    return pd.to_datetime(int(timestamp_integer / 1000), utc=True, unit='s')

def transform_to_timestamp_integer(datetime_object):
    '''
    As data points involved milliseconds, we need to transform them by constant 1000.
    '''

    return int(datetime_object.timestamp() * 1000)

def obtain_full_spotdata(start_timestamp,
                         end_timestamp,
                         exchange, symbol, interval = '1h',
                         limit = 1000):

    time_difference = int(convert_to_seconds(interval) * limit * 1000)

    full_data_list = []

    curr_time = start_timestamp + time_difference
    while (curr_time + time_difference < end_timestamp):
        data_list = get_spot(exchange = exchange, symbol = symbol, interval = interval,
                             endTime = curr_time,
                             limit = limit)
        full_data_list = full_data_list + data_list

        time.sleep(0.2)
        curr_time += time_difference

    data_list = get_spot(exchange = exchange, symbol = symbol, interval = interval,
                        startTime = curr_time,
                        endTime = end_timestamp,
                        limit = limit)

    full_data_list = full_data_list + data_list

    return full_data_list

if __name__ == "__main__":
    test = download_data('2015-01-01', '2025-02-24', interval='1d', filepath='../data/raw/', filename='test', save=True)
    print(test)