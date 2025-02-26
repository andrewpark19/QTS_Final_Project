# pip install --upgrade coinapi_rest_v1

from coinapi_rest_v1.restapi import CoinAPIv1
import datetime, sys
import polars as pl
from config import API_KEY, EXCHANGE_ID, STABLE_COINS

api = CoinAPIv1(API_KEY)

def get_symbols(exchange_id=EXCHANGE_ID):
    
    symbols = api.metadata_list_symbols(exchange_id=exchange_id)
    symbols_df = pl.DataFrame(symbols)
    return symbols_df

def get_crypto_universe(symbols_df, start_date, end_date, base_curr = 'USDT', min_daily_vol=1e6):
    '''
    symbols_df: polars.DataFrame
    start_date: str, format: 'YYYY-MM-DD'
    end_date: str, format: 'YYYY-MM-DD'
    base_curr: str, default: 'USDT'; can be 'USD'
    min_daily_vol: float, default: 1e6
    '''
    filtered_df = symbols_df.filter((pl.col("asset_id_quote") == base_curr) &
                                    (~pl.col("asset_id_base").is_in(STABLE_COINS)) &
                                    (pl.col("data_start") <= start_date) &
                                    (pl.col("data_end") >= end_date) &
                                    (pl.col("volume_1day_usd") > min_daily_vol))

    crypto_universe = filtered_df.select('symbol_id').unique()['symbol_id']
    min_date = filtered_df['data_start'].max()
    max_date = filtered_df['data_end'].min()
    
    return crypto_universe, min_date, max_date

def download_data(start, end, interval, tickers):
    
    all_data = {}
    
    for ticker in tickers:
        ohlcv_historical = api.ohlcv_historical_data(ticker, {'period_id': interval, 'time_start': start, 'time_end': end, 'limit': 10000})
        all_data[ticker] = ohlcv_historical
        
    return all_data


def get_historical_data(start_date, end_date, interval='1DAY', base_curr='USDT', min_daily_vol=1e6, exchange_id=EXCHANGE_ID, filepath='../data/raw/', filename=None, save=True):
    
    symbols_df = get_symbols(exchange_id)
    crypto_universe, min_date, max_date = get_crypto_universe(symbols_df, start_date, end_date, base_curr, min_daily_vol)
    all_data = download_data(min_date, max_date, interval, crypto_universe)
    
    pl_dfs = []
    
    
    for key in all_data:
        pl_df = pl.DataFrame(all_data[key])
        pl_df = pl_df.with_columns(
            pl.lit(key).alias('symbol_id'),
            (pl.col('price_close') * pl.col('volume_traded')).alias('calculated_vol_usd')
        )
        pl_dfs.append(pl_df)
    res = pl.concat(pl_dfs)
    
    if filename is None:
        filename = f'raw_crypto_data_{min_date}_{max_date}_{interval}.parquet'
    FILEPATH = filepath + filename
    
    if save:
        res.write_parquet(f'../data/raw/raw_crypto_data_{min_date}_to_{max_date}_{interval}.parquet')
        print(f'Data saved to {FILEPATH}')
    return res
        
    
if __name__ == '__main__':
    start = datetime.date(2018, 1, 1).isoformat()
    end = datetime.date(2025, 2, 24).isoformat()
    interval = '1DAY'
    
    ohlc_data = get_historical_data(start, end, interval)
    # display(ohlc_data)
    # display(ohlc_data['symbol_id'].unique())