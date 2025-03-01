import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from datetime import datetime

def scrape_market_cap(date, tickers):
    """
    Scrape market ap data for a given snapshot date from CoinMarketCap.
    'date' should be in 'YYYYMMDD' format.
    'tickers' is a list of ticker symbols to filter.
    """
    url = f'https://coinmarketcap.com/historical/{date}/'
    headers = {'User-Agent': 'Mozilla/5.0'}  # Use a proper user-agent to avoid request issues
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve data for {date}")

    soup = BeautifulSoup(response.content, 'html.parser')
    soup_str = str(soup)
    # Define your markers (adjust if necessary)
    start_marker = '{\\"data\\":['
    end_marker = '}}}]'
    # Find the start and end positions
    start_index = soup_str.find(start_marker)
    end_index = soup_str.find(end_marker, start_index)
    
    json_str = soup_str[start_index:end_index+len(end_marker)] + '}'
    fixed_str = json_str.encode('utf-8').decode('unicode_escape')
    data_dict = json.loads(fixed_str)

    df = pd.DataFrame(data_dict['data'])
    df['Market Cap (USD)'] = pd.DataFrame(pd.DataFrame(df['quote'].values.tolist())['USD'].values.tolist())['marketCap']
    filtered_df = df[df['symbol'].isin(tickers)][['symbol', 'Market Cap (USD)']]
    filtered_df['date'] = date

    return filtered_df

def scrape_cmc(start_date, end_date, tickers, save=True, filename=None, filepath='data/raw/'):
    
    dates = pd.date_range(start=start_date, end=end_date, freq='W-SUN').strftime('%Y%m%d').tolist()
    all_data = pd.DataFrame()
    for date in tqdm(dates):
        try:
            data = scrape_market_cap(date, tickers)
        except:
            print('Error trying to scrape data')
            data = pd.DataFrame()
        all_data = pd.concat([all_data, data])
    
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data = all_data.pivot(index='date', columns='symbol', values='Market Cap (USD)')
    
    if save:
        if not filename:
            print('No Filename given, will save as current time')
            # Get the current date and time
            now = datetime.now()
            # Format the date and time as a timestamp
            filename = now.strftime("%Y-%m-%d_%H%M")
        FILEPATH = filepath+filename+'.parquet'
        all_data.to_parquet(FILEPATH)
        
    return all_data

# Example usage:
if __name__ == "__main__":
    # Define the list of snapshot dates and tickers of interest.
    start_date = '20190923'
    end_date = '20200225'
    tickers = ['BTC', 'ETH', 'XRP', 'DOGE', 'BAT']     # Example tickers
    df = scrape_cmc(start_date, end_date, tickers, filename='test', filepath='../data/raw/')
    
    

