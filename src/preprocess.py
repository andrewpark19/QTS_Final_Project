import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf


def pivot_value(data, value_col='Close'):
    data = data.with_columns(
        # Convert "Close time" from string (milliseconds) to datetime (multiplying by 1000)
        ((pl.col("Close time").cast(pl.Int64) * 1000).cast(pl.Datetime)).alias("close_time"),
        # Convert "Close" from string to float
        pl.col(value_col).cast(pl.Float64).alias(value_col)
    )
    
    # Create a new column that extracts just the date part
    data = data.with_columns(
        pl.col("close_time").dt.date().alias("date")
    )
    
    # Optionally drop the original "close_time" if you no longer need it
    data = data.drop("close_time")
    
    # Pivot using the new "date" column as the index
    pivoted_data = data.pivot(values=value_col, index="date", on="Ticker")
    
    # Rename ticker columns to remove 'USDT'
    new_names = {col: col.replace('USDT', '') for col in pivoted_data.columns if col != "date"}
    pivoted_data = pivoted_data.rename(new_names)
    
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

def get_pct_change_resampled(df: pl.DataFrame, freq: str = "1d", keep_dates: bool = True, dropna: bool = True) -> pl.DataFrame:
    """
    Resamples daily data to the specified frequency and calculates the percentage change on the resampled data.
    
    Parameters:
        df (pl.DataFrame): Input Polars DataFrame with a "date" column.
        freq (str): Resampling frequency (e.g., "3d" for 3 days, "1w" for one week, "1mo" for one month). Defaults to "1d" (daily).
        keep_dates (bool): Whether to keep the date column in the returned DataFrame.
        dropna (bool): Whether to drop rows with null values (which may appear from pct_change).
    
    Returns:
        pl.DataFrame: A Polars DataFrame with the percentage change computed on the resampled data.
    """
    # Convert the Polars DataFrame to a Pandas DataFrame.
    pdf = df.to_pandas()
    
    # Ensure the "date" column is a datetime type.
    pdf['date'] = pd.to_datetime(pdf['date'])
    
    # Set the "date" column as the index for resampling.
    pdf.set_index('date', inplace=True)
    
    # Resample the DataFrame using the provided frequency and take the last observation in each period.
    pdf_resampled = pdf.resample(freq).last()
    
    # Calculate the percentage change on each column.
    pct_change_pdf = pdf_resampled.pct_change()
    
    # Optionally drop rows with NA values.
    if dropna:
        pct_change_pdf = pct_change_pdf.dropna()
    
    # Optionally reset the index to include the date column.
    if keep_dates:
        pct_change_pdf = pct_change_pdf.reset_index()
    
    pct_change_pdf = pl.from_pandas(pct_change_pdf)
    if isinstance(pct_change_pdf['date'].dtype, pl.datatypes.Datetime):
        pct_change_pdf = pct_change_pdf.with_columns(pl.col('date').dt.date().alias('date'))
    
    # Convert the resulting Pandas DataFrame back to a Polars DataFrame.
    return pct_change_pdf

def get_corr_matrix(df, plot=True):
    min_date = df['date'].min()
    max_date = df['date'].max()
    pd_df = df.drop('date').to_pandas()
    corr_matrix = pd_df.corr()

    if plot:
        # Set figure size and style
        plt.figure(figsize=(25, 10))
        sns.set_theme(style="whitegrid")
        
        # Create heatmap with enhanced aesthetics
        ax = sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.75,
            annot_kws={"size": 15},  # Adjust annotation font size
            cbar_kws={"shrink": 0.75, "aspect": 40}  # Adjust color bar size
        )

        # Add a more prominent title
        plt.title(
            f"Cryptocurrency Return Correlation Matrix \n({min_date} to {max_date})", 
            fontsize=16, fontweight="bold", pad=20
        )

        # Improve tick label visibility
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right", fontsize=12)

        # Show the plot
        plt.show()
    
    return corr_matrix


def plot_acf_pacf(pl_df: pl.DataFrame, tickers: list[str], nlags: int = 40, include_0: bool = False, freq_title='Daily'):
    # Set a seaborn theme for aesthetic plots
    sns.set_theme(style="whitegrid")
    
    # Convert the Polars DataFrame to Pandas and set date as index
    df = pl_df.to_pandas()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    
    num_tickers = len(tickers)
    fig, axes = plt.subplots(nrows=num_tickers, ncols=2, figsize=(14, 4 * num_tickers))
    
    # Ensure axes is 2D even if only one ticker is provided
    if num_tickers == 1:
        axes = [axes]
    
    for i, ticker in enumerate(tickers):
        series = df[ticker].dropna()
        N = len(series)
        # 95% confidence interval for zero autocorrelation
        conf_int = 1.96 / np.sqrt(N)
        
        # Compute ACF and PACF
        lag_acf = acf(series, nlags=nlags)
        lag_pacf = pacf(series, nlags=nlags)
        lags = np.arange(len(lag_acf))
        
        # If include_0 is False, remove the 0th lag
        if not include_0:
            lags = lags[1:]
            lag_acf = lag_acf[1:]
            lag_pacf = lag_pacf[1:]
        
        # Plot ACF using vertical lines
        ax1 = axes[i][0]
        ax1.vlines(x=lags, ymin=0, ymax=lag_acf, color='b')
        ax1.scatter(lags, lag_acf, color='b', zorder=3)
        ax1.axhline(0, color='black', lw=1)
        ax1.axhline(conf_int, color='red', linestyle='--', lw=1)
        ax1.axhline(-conf_int, color='red', linestyle='--', lw=1)
        ax1.set_title(f'{ticker} - ACF - {freq_title} returns \n ({start_date} to {end_date})')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('ACF')
        
        # Plot PACF using vertical lines
        ax2 = axes[i][1]
        ax2.vlines(x=lags, ymin=0, ymax=lag_pacf, color='b')
        ax2.scatter(lags, lag_pacf, color='b', zorder=3)
        ax2.axhline(0, color='black', lw=1)
        ax2.axhline(conf_int, color='red', linestyle='--', lw=1)
        ax2.axhline(-conf_int, color='red', linestyle='--', lw=1)
        ax2.set_title(f'{ticker} - PACF - {freq_title} returns \n ({start_date} to {end_date})')

        ax2.set_xlabel('Lag')
        ax2.set_ylabel('PACF')
    
    plt.tight_layout()
    plt.show()

def run_ar_regression(pl_df: pl.DataFrame, tickers: list[str], p: int):
    """
    Runs an AR(p) regression for each ticker in the provided Polars DataFrame.
    
    The model is:
        y[t] = beta0 + beta1 * y[t-1] + ... + beta_p * y[t-p] + error
    
    Parameters:
        pl_df (pl.DataFrame): A Polars DataFrame that includes a 'date' column and return data columns.
        tickers (list[str]): A list of ticker names (which are column names in the DataFrame) to run the regression on.
        p (int): The order of the autoregressive model (number of lags).
        
    Returns:
        dict: A dictionary keyed by ticker. For each ticker, returns a dictionary with:
            - "coefficients": dict of regression coefficients (including constant)
            - "p_values": dict of p-values for each coefficient.
    """
    # Convert the Polars DataFrame to Pandas and ensure 'date' is datetime
    pdf = pl_df.to_pandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.sort_values('date')
    
    if not tickers:
        tickers = pdf.columns[1:]
    
    results = {}
    
    for ticker in tickers:
        series = pdf[ticker]
        # Create a DataFrame for regression: y[t] as the dependent variable and y[t-1]...y[t-p] as predictors.
        df_reg = pd.DataFrame({'y': series})
        # Create lag columns for lags 1 to p.
        for lag in range(1, p + 1):
            df_reg[f'lag_{lag}'] = series.shift(lag)
        # Drop rows with NaN values (due to lagging)
        df_reg = df_reg.dropna()
        
        # Define predictors (with constant) and dependent variable.
        X = df_reg[[f'lag_{lag}' for lag in range(1, p + 1)]]
        X = sm.add_constant(X)  # add constant term (beta0)
        y = df_reg['y']
        
        # Run the OLS regression
        model = sm.OLS(y, X).fit()
        
        # Store coefficients and p-values
        results[ticker] = {
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict()
        }
    
    return results

def extract_ar_params(results: dict, lags: list[int], concat=False):
    """
    Extracts AR regression parameters for the specified lags from the results dictionary.
    
    Parameters:
        results (dict): Dictionary with tickers as keys and each value a dict with "coefficients" and "p_values".
        lags (list[int]): List of lags to extract (e.g., [1,2] to extract 'lag_1' and 'lag_2').
    
    Returns:
        tuple: Two Polars DataFrames. The first contains the coefficients for each ticker for the specified lags,
               and the second contains the p-values.
               
        Both DataFrames have a "Ticker" column and additional columns "lag_1", "lag_2", etc.
    """
    coeffs_list = []
    pvals_list = []
    
    for ticker, res in results.items():
        # Create a dictionary for the ticker with its name.
        coeff_entry = {"Ticker": ticker}
        pval_entry = {"Ticker": ticker}
        
        # Loop through each specified lag and extract if available.
        for lag in lags:
            lag_key = f"lag_{lag}"
            if lag_key in res["coefficients"]:
                coeff_entry[lag_key] = res["coefficients"][lag_key]
            else:
                coeff_entry[lag_key] = None
            if lag_key in res["p_values"]:
                pval_entry[lag_key] = res["p_values"][lag_key]
            else:
                pval_entry[lag_key] = None
        
        coeffs_list.append(coeff_entry)
        pvals_list.append(pval_entry)
    
    coeffs_df = pl.DataFrame(coeffs_list)
    pvals_df = pl.DataFrame(pvals_list)
    
    if not concat:
        concat_df = (coeffs_df, pvals_df)
    else:
        # Rename all columns except "Ticker" with a 'coeff_' prefix.
        coeffs_df_prefixed = coeffs_df.rename({
            col: f"coeff_{col}" for col in coeffs_df.columns if col != "Ticker"
        })

        # Rename all columns except "Ticker" with a 'pval_' prefix.
        pvals_df_prefixed = pvals_df.rename({
            col: f"pval_{col}" for col in pvals_df.columns if col != "Ticker"
        })

        # Option 1: Use join on the common "Ticker" column.
        concat_df = coeffs_df_prefixed.join(pvals_df_prefixed, on="Ticker", how="inner")
        
    return concat_df

def plot_dist(pl_df, lag, label, freq, min_date, max_date, kde=False, bins=20):
    '''
    lag: int, the lag number to plot
    label: str, 'Betas' or 'P-Values'
    freq: str, Frequency of returns used
    '''
    if label != 'Betas' and label != 'P-Values':
        raise ValueError('Warning: Expecting label to be either "Betas" or "P-Values"')
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")

    # Create the histogram with KDE overlay
    plt.figure(figsize=(15, 5))
    sns.histplot(pl_df[f'lag_{lag}'], bins=bins, kde=kde, color='royalblue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel(f"Lag-{lag} {label}", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of Lag-{lag} {label} for Cryptocurrency {freq} Returns ({min_date} to {max_date})", fontsize=14, fontweight='bold')

    # Add a vertical line at 0.05 significance threshold
    if label == 'P-Values':
        plt.axvline(0.05, color='red', linestyle='dashed', linewidth=2, label="0.05 Significance Level")
    else:
        plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label="Zero")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# if __name__ == '__main__':
    
    # from data_binance import download_data
    
    # start_date = '2015-01-01'
    # end_date = '2025-02-24'
    # interval='1d'
    # filepath = 'data/raw/'
    # raw_data = pl.from_pandas(download_data(start_date, end_date, interval, filepath=filepath))
    # test = pivot_value(raw_data)

    
    
    