import polars as pl
import matplotlib.pyplot as plt
import datetime


def pivot_close_price(data):
    data = data.with_columns(
        # Convert "Close time" from string (milliseconds) to datetime (multiplying by 1000)
        ((pl.col("Close time").cast(pl.Int64) * 1000).cast(pl.Datetime)).alias("close_time"),
        # Convert "Close" from string to float
        pl.col("Close").cast(pl.Float64).alias("Close")
    )
    
    # Create a new column that extracts just the date part
    data = data.with_columns(
        pl.col("close_time").dt.date().alias("date")
    )
    
    # Optionally drop the original "close_time" if you no longer need it
    data = data.drop("close_time")
    
    # Pivot using the new "date" column as the index
    pivoted_data = data.pivot(values="Close", index="date", on="Ticker")
    return pivoted_data

def plot_valid_tickers(df):
    df_clone = df.clone()
    df_clone = df_clone.with_columns(
        pl.fold(
            acc=pl.lit(0),
            function=lambda acc, col: acc + col.is_not_null().cast(pl.Int64),
            # Exclude the date column from the count:
            exprs=pl.all().exclude("date")
        ).alias("non_null_count")
    ).with_columns(
        pl.col('non_null_count').pct_change().alias('pct_change')
    )
    
    # Select only the date and non-null count columns and convert to pandas
    df_plot = df_clone.select(["date", "non_null_count"]).to_pandas()

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["date"], df_plot["non_null_count"])
    plt.xlabel("Date")
    plt.ylabel("Non-Null Count")
    plt.title("Non-Null Count with Respect to Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df_clone

def filter_date(close_prices, start_date):
    if isinstance(start_date, str):
        start_date = datetime.date.fromisoformat('2019-11-15')
        
    close_prices = close_prices.filter(pl.col('date')>=start_date)
    
    nc = close_prices.null_count().row(0)
    cols_to_keep = [col for col, count in zip(close_prices.columns, nc) if count == 0]
    close_prices = close_prices.select(cols_to_keep)
    
    return close_prices

if __name__ == '__main__':
    
    from data_binance import download_data
    
    start_date = '2015-01-01'
    end_date = '2025-02-24'
    interval='1d'
    filepath = 'data/raw/'
    raw_data = pl.from_pandas(download_data(start_date, end_date, interval, filepath=filepath))
    test = pivot_close_price(raw_data)