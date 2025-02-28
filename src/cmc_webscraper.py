# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import re

# def scrape_market_cap(date, tickers):
#     """
#     Scrape market ap data for a given snapshot date from CoinMarketCap.
#     'date' should be in 'YYYYMMDD' format.
#     'tickers' is a list of ticker symbols to filter.
#     """
#     url = f'https://coinmarketcap.com/historical/{date}/'
#     headers = {'User-Agent': 'Mozilla/5.0'}  # Use a proper user-agent to avoid request issues
#     response = requests.get(url, headers=headers)
    
#     if response.status_code != 200:
#         print(f"Failed to retrieve data for {date}")
#         return []

#     soup = BeautifulSoup(response.content, 'html.parser')
    
#     # Locate the table containing the data.
#     # (You might need to adjust this selector based on the page structure.)
#     table = soup.find('table')
#     if table is None:
#         print("Could not find the data table on the page.")
#         return []
    
#     rows = table.find_all('tr')
#     results = []

#     for row in rows[1:]:  # Skip the header row
#         cols = row.find_all('td')
#         if not cols:
#             continue
        
#         # Assume the first few columns contain the symbol and market cap.
#         # You may need to adjust indices depending on the actual structure.
#         try:
#             # Example: The first column might be the rank, the second the symbol.
#             symbol = cols[1].get_text(strip=True)
#         except IndexError:
#             continue
        
#         # Only proceed if the symbol is one we're interested in
#         if symbol in tickers:
#             market_cap = None
#             # Look for a cell that looks like a market cap (starts with '$' and contains commas)
#             for col in cols:
#                 text = col.get_text(strip=True)
#                 if text.startswith('$') and ',' in text:
#                     market_cap = text
#                     break
            
#             results.append({
#                 'date': date,
#                 'ticker': symbol,
#                 'market_cap': market_cap
#             })
#     return results

# def scrape_data(dates, tickers):
#     all_data = []
#     for date in dates:
#         data = scrape_market_cap(date, tickers)
#         all_data.extend(data)
#     return pd.DataFrame(all_data)

# # Example usage:
# if __name__ == "__main__":
#     # Define the list of snapshot dates and tickers of interest.
#     dates = ['20191103', '20191110']  # Example dates in YYYYMMDD format
#     tickers = ['BTC', 'ETH', 'XRP']     # Example tickers
#     df = scrape_data(dates, tickers)
#     print(df)

import requests
from bs4 import BeautifulSoup

url = 'https://coinmarketcap.com/historical/20191103/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/115.0.0.0 Safari/537.36'
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    # For example, the coin rows might be within table rows.
    # (You may need to inspect the HTML to adjust the selectors.)
    rows = soup.select('table tbody tr')
    for row in rows:
        # Get the text from each cell and join them with a separator
        cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
        print(" | ".join(cells))
else:
    print("Error: ", response.status_code)
