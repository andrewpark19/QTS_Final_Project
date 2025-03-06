import polars as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------------
# Helper Function: Percentile Rank
# -------------------------------
def compute_percentile_rank_series(series, interval: int = None, window: int = None, min_periods=None):
    """
    Compute the percentile rank for a numeric series.
    
    For each index i, the percentile rank is computed using a subset of historical returns:
      - If `interval` is provided as an integer, we select values at indices:
            i, i-interval, i-2*interval, ...
        If `interval` is None, then we use every index.
      - If `window` is None (default), the lookback is expanding (i.e. all available valid observations are used).
        If `window` is an integer, only the most recent `window` observations (based on the sampling determined by interval) are used.
      - The percentile rank at index i is defined as:
        
            rank = (number of valid values in the selected window that are <= current value)
                   / (total number of valid values in the selected window)
        
    If min_periods is specified, then if the number of valid observations is less than min_periods,
    the percentile rank is set to None.
    """
    n = len(series)
    ranks = [None] * n

    for i in range(n):
        current_value = series[i]
        
        # Check if current value is valid.
        if current_value is None or (isinstance(current_value, float) and np.isnan(current_value)):
            ranks[i] = None
            continue
        
        # Determine indices to include.
        if interval is None:
            # Use every index.
            start_idx = 0 if window is None else max(0, i - window + 1)
            indices = list(range(start_idx, i + 1))
        else:
            # For an interval-based selection, we require at least one prior value at the given interval.
            if i < interval:
                ranks[i] = None
                continue
            # Build indices: i, i-interval, i-2*interval, ... until index >= 0.
            indices = list(range(i, -1, -interval))
            if window is not None:
                # Only take the most recent "window" many indices.
                indices = indices[:window]
        
        # Filter the window to only valid (non-NaN) values.
        valid_values = []
        for idx in indices:
            val = series[idx]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                valid_values.append(val)
        
        if min_periods is not None and len(valid_values) < min_periods:
            ranks[i] = None
        elif not valid_values:
            ranks[i] = None
        else:
            count = sum(1 for x in valid_values if x <= current_value)
            percentile = count / len(valid_values)
            ranks[i] = percentile

    return ranks

# -------------------------------
# Updated Helper Functions for Percentile Ranks
# -------------------------------
def add_prev_return_percentile_ranks(prices, weights, j, window=None, min_periods=None):
    """
    Computes the percentile ranks for the j-day previous log return.
    
    If weights is provided, it computes the portfolio return as a weighted sum of individual asset returns.
    If weights is None, prices is assumed to contain only one ticker.
    Returns a DataFrame with a single column 'prev_p' and the same index as prices.
    """
    # Compute j-day log returns: r_{t-j,t} = ln(P_t / P_{t-j})
    log_returns = np.log(prices / prices.shift(j))
    
    if weights is not None:
        if weights.shape != prices.shape:
            raise ValueError("Weights must have the same shape as prices.")
        # Compute portfolio return as the weighted sum across tickers.
        port_return = (log_returns * weights).sum(axis=1)
    else:
        if prices.shape[1] != 1:
            raise ValueError("When weights is None, prices should have a single ticker.")
        port_return = log_returns.iloc[:, 0]
    
    # Convert the return series to list and compute percentile ranks.
    ret_list = port_return.tolist()
    percentiles = compute_percentile_rank_series(ret_list, interval=j, window=window, min_periods=min_periods)
    return pd.DataFrame({'prev_p': percentiles}, index=prices.index)

def add_lookahead_return_percentile_ranks(prices, weights, k, window=None, min_periods=None):
    """
    Computes the percentile ranks for the k-day look-ahead log return.
    
    If weights is provided, it computes the portfolio look-ahead return as a weighted sum.
    If weights is None, prices is assumed to contain only one ticker.
    Returns a DataFrame with a single column 'lookahead_p' and the same index as prices.
    """
    # Compute k-day lookahead log returns: r_{t,t+k} = ln(P_{t+k} / P_t)
    lookahead_returns = np.log(prices.shift(-k) / prices)
    
    if weights is not None:
        if weights.shape != prices.shape:
            raise ValueError("Weights must have the same shape as prices.")
        port_return = (lookahead_returns * weights).sum(axis=1)
    else:
        if prices.shape[1] != 1:
            raise ValueError("When weights is None, prices should have a single ticker.")
        port_return = lookahead_returns.iloc[:, 0]
    
    ret_list = port_return.tolist()
    percentiles = compute_percentile_rank_series(ret_list, interval=k, window=window, min_periods=min_periods)
    return pd.DataFrame({'lookahead_p': percentiles}, index=prices.index)

# -------------------------------
# Regression Function
# -------------------------------
def regression_percentile_ranks(prices, weights, j_list, k_list, window=None, min_periods=None):
    """
    For each combination of previous lookback j in j_list and look-ahead holding period k in k_list,
    computes the percentile ranks for the j-day previous log return and k-day lookahead log return.
    Then runs the regression:
    
         lookahead_p = alpha + beta * prev_p + epsilon
    
    For each j and k combination, saves the beta coefficient, its p-value, and the R^2 of the regression.
    
    Parameters:
      prices: pd.DataFrame of prices (dates as index, tickers as columns).
      weights: pd.DataFrame of weights (same shape as prices) or None (in which case prices should have one column).
      j_list: list of integers (previous lookback periods).
      k_list: list of integers (look-ahead holding periods).
      window: window parameter to be passed to the helper functions.
      min_periods: minimum valid observations required to compute percentile rank.
    
    Returns:
      Three DataFrames: beta_df, pval_df, r2_df. The rows are labelled by j and columns by k.
    """
    # Initialize empty DataFrames to store results.
    beta_df = pd.DataFrame(index=j_list, columns=k_list, dtype=float)
    pval_df = pd.DataFrame(index=j_list, columns=k_list, dtype=float)
    r2_df = pd.DataFrame(index=j_list, columns=k_list, dtype=float)
    
    for j in tqdm(j_list, desc="Processing j values"):
        # Compute previous percentile ranks for j-day return.
        prev_df = add_prev_return_percentile_ranks(prices, weights, j, window=window, min_periods=min_periods)
        for k in tqdm(k_list, desc="Processing k values", leave=False):
            # Compute lookahead percentile ranks for k-day return.
            lookahead_df = add_lookahead_return_percentile_ranks(prices, weights, k, window=window, min_periods=min_periods)
            # Merge the two series on the date index.
            reg_df = prev_df.join(lookahead_df)
            reg_df = reg_df.dropna()
            if len(reg_df) == 0:
                beta_df.loc[j, k] = np.nan
                pval_df.loc[j, k] = np.nan
                r2_df.loc[j, k] = np.nan
                continue
            
            # Run the regression: lookahead_p = alpha + beta * prev_p + epsilon
            X = reg_df['prev_p']
            y = reg_df['lookahead_p']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            beta_df.loc[j, k] = model.params['prev_p']
            pval_df.loc[j, k] = model.pvalues['prev_p']
            r2_df.loc[j, k] = model.rsquared
    
    return beta_df, pval_df, r2_df

# -------------------------------
# Plotting Function
# -------------------------------
def plot_heatmaps(beta_df: pd.DataFrame, pval_df: pd.DataFrame, r2_df: pd.DataFrame):
    """
    Given three DataFrames:
       - beta_df: rows indexed by j values and columns labeled by k values.
       - pval_df: same structure containing p-values.
       - r2_df: same structure containing R² values.
    This function creates three separate heatmaps (using matplotlib):
       one for beta coefficients, one for p-values, and one for R² values.
    """
    # Set up a common plotting style.
    # plt.style.use('seaborn-whitegrid')
    
    # Plot Beta heatmap.
    plt.figure(figsize=(8, 6))
    plt.imshow(beta_df.values, aspect='auto', cmap='viridis')
    plt.colorbar(label='Beta')
    plt.xticks(ticks=np.arange(len(beta_df.columns)), labels=beta_df.columns)
    plt.yticks(ticks=np.arange(len(beta_df.index)), labels=beta_df.index)
    plt.title("Beta Heatmap (rows: j, cols: k)")
    plt.xlabel("k (look-ahead holding period)")
    plt.ylabel("j (previous lookback)")
    plt.show()
    
    # Plot P-Value heatmap.
    plt.figure(figsize=(8, 6))
    plt.imshow(pval_df.values, aspect='auto', cmap='viridis')
    plt.colorbar(label='P-Value')
    plt.xticks(ticks=np.arange(len(pval_df.columns)), labels=pval_df.columns)
    plt.yticks(ticks=np.arange(len(pval_df.index)), labels=pval_df.index)
    plt.title("P-Value Heatmap (rows: j, cols: k)")
    plt.xlabel("k (look-ahead holding period)")
    plt.ylabel("j (previous lookback)")
    plt.show()
    
    # Plot R² heatmap.
    plt.figure(figsize=(8, 6))
    plt.imshow(r2_df.values, aspect='auto', cmap='viridis')
    plt.colorbar(label='R²')
    plt.xticks(ticks=np.arange(len(r2_df.columns)), labels=r2_df.columns)
    plt.yticks(ticks=np.arange(len(r2_df.index)), labels=r2_df.index)
    plt.title("R² Heatmap (rows: j, cols: k)")
    plt.xlabel("k (look-ahead holding period)")
    plt.ylabel("j (previous lookback)")
    plt.show()


def equal_weight_portfolio(df: pl.DataFrame, to_pd=True):
    """
    Given a Polars DataFrame of returns (with a 'date' column and one column per coin),
    returns two DataFrames:
    
    1. A portfolio return DataFrame containing:
         - date: the date column,
         - EW_return: the equal weighted portfolio return for that day,
           computed as the average of the coin returns.
           
    2. A weight DataFrame containing:
         - date: the date column,
         - For each coin column, a column with the constant equal weight for that day.
           (All coins have equal weight = 1 / number_of_coins)
    
    Parameters:
      df: Polars DataFrame. The first column must be 'date'; the remaining columns are coin returns.
    
    Returns:
      portfolio_df: A DataFrame with the date and equal weighted portfolio return.
      weights_df: A DataFrame with the date and the weight of each coin on that day.
    """
    # Get the list of coin columns (all columns except "date")
    coin_cols = [col for col in df.columns if col != "date"]
    
    n_coins = len(coin_cols)
    if n_coins == 0:
        raise ValueError("DataFrame must contain at least one coin column besides 'date'.")
        
    # Compute the equal-weighted portfolio return: average return across coins
    portfolio_df = df.with_columns(
        (sum([pl.col(coin) for coin in coin_cols]) / n_coins).alias("EW_return")
    ).select(["date", "EW_return"])
    
    # Create a weights DataFrame.
    # For an equal weighted portfolio, each coin has constant weight = 1 / n_coins for every day.
    # We can create a column for each coin with this constant value.
    weights_exprs = [pl.lit(1/n_coins).alias(coin) for coin in coin_cols]
    weights_df = df.select(["date"]).with_columns(weights_exprs)
    
    if to_pd:
      portfolio_df = portfolio_df.to_pandas()
      portfolio_df.set_index('date', inplace=True)
      weights_df = weights_df.to_pandas()
      weights_df.set_index('date', inplace=True)
    return portfolio_df, weights_df

def marketcap_weighted_portfolio(return_df: pl.DataFrame, mcap_df: pl.DataFrame, to_pd=True):
    """
    Given:
      - return_df: A Polars DataFrame of daily returns with a "date" column and one column per coin.
      - mcap_df: A Polars DataFrame of weekly market caps with a "date" column and one column per coin.
    
    This function:
      1. Performs an asof join using a backward strategy to attach the most recent (historical)
         market cap to each daily return. We use a suffix "_mcap" for the market cap columns.
      2. Fills tail-end nulls (i.e. for dates after the last weekly market cap) using a backward fill.
      3. Identifies coins common to both DataFrames.
      4. Computes, for each day, each coin’s weight as (coin_mcap / total_mcap).
      5. Computes the market-cap weighted portfolio return as the sum over coins of (weight * return).
    
    Returns:
      - portfolio_df: A DataFrame with "date" and the market cap weighted portfolio return (column "MC_weighted_return").
      - weights_df: A DataFrame with "date" and one column per coin (named "w_<coin>") containing the coin's weight.
    """
    # Ensure both DataFrames are sorted by date.
    return_df = return_df.sort("date")
    mcap_df = mcap_df.sort("date")
    
    # Perform an asof join using a backward strategy and a suffix for market cap columns.
    # For each daily return, attach the most recent historical market cap.
    daily = return_df.join_asof(mcap_df, on="date", strategy="backward", suffix="_mcap")
    
    # For dates after the last weekly market cap, the mcap columns will be null.
    # Fill those nulls by a backward fill (i.e. use the last available market cap).
    daily = daily.fill_null(strategy="backward")
    
    # Identify coins common to both DataFrames.
    # For returns, the coin columns are in return_df (exclude "date").
    # For market cap, they appear as "<coin>_mcap" in the joined DataFrame.
    ret_coins = set(return_df.columns) - {"date"}
    mcap_coins = {col.replace("_mcap", "") for col in daily.columns if col.endswith("_mcap")}
    common_coins = sorted(ret_coins.intersection(mcap_coins))
    
    if not common_coins:
        raise ValueError("No common coin columns found between the returns and market cap DataFrames.")
    
    # Compute total market cap using the market cap columns.
    # We refer to the mcap columns using the suffix "_mcap".
    total_mcap_expr = sum(pl.col(f"{coin}_mcap") for coin in common_coins)
    daily = daily.with_columns(total_mcap_expr.alias("total_mcap"))
    
    # Compute weights for each coin: weight = coin's market cap / total market cap.
    for coin in common_coins:
        daily = daily.with_columns(
            (pl.col(f"{coin}_mcap") / pl.col("total_mcap")).alias(f"{coin}")
        )
    
    # Compute the market-cap weighted portfolio return.
    # For each coin, multiply its return (from the return_df) by its weight and sum.
    portfolio_expr = sum(pl.col(f"{coin}") * pl.col(coin) for coin in common_coins)
    portfolio_df = daily.with_columns(
        portfolio_expr.alias("MCW_return")
    ).select(["date", "MCW_return"])
    
    # Create a weights DataFrame with date and each coin's weight.
    weight_cols = [f"{coin}" for coin in common_coins]
    weights_df = daily.select(["date"] + weight_cols)
    
    if to_pd:
      portfolio_df = portfolio_df.to_pandas()
      portfolio_df.set_index('date', inplace=True)
      weights_df = weights_df.to_pandas()
      weights_df.set_index('date', inplace=True)
    return portfolio_df, weights_df

# -------------------------------
# Signal Generation
# -------------------------------

def get_signals(price_df, weights, j, p, long_short='long-short', style='momentum', min_periods=None, window=None):
    # Validate input dimensions.
    if weights is None:
        if price_df.shape[1] != 1:
            raise ValueError("Weights must be provided when multiple tickers are present.")
    else:
        if weights.shape != price_df.shape:
            raise ValueError("Weights must have the same shape as price_df.")
    
    # Compute the j-day log returns for each asset.
    log_returns = np.log(price_df / price_df.shift(j))
    
    # Form portfolio return: either weighted sum or (if only one asset) the single return series.
    if weights is not None:
        port_return = (log_returns * weights).sum(axis=1)
    else:
        port_return = log_returns.iloc[:, 0]
    
    # Convert the portfolio return series to a list.
    ret_list = port_return.tolist()
    
    # Compute the historical percentile rank for each j-day return using the provided helper.
    # Here we use j as the interval (non-overlapping returns), with the optional window and min_periods.
    percentiles = compute_percentile_rank_series(ret_list, interval=j, window=window, min_periods=min_periods)
    percentiles_series = pd.Series(percentiles, index=port_return.index)
    
    # Define a helper to generate signals from a percentile rank.
    def generate_signal(rank_val):
        # Treat missing rank as a neutral signal.
        if rank_val is None or (isinstance(rank_val, float) and np.isnan(rank_val)):
            return 0
        if style == 'momentum':
            if long_short == 'long-short':
                if rank_val > (1 - p):
                    return 1
                elif rank_val < p:
                    return -1
                else:
                    return 0
            elif long_short == 'long':
                return 1 if rank_val > (1 - p) else 0
            elif long_short == 'short':
                return -1 if rank_val < p else 0
            else:
                raise ValueError("Invalid long_short value. Must be one of 'long-short', 'long', or 'short'.")
        elif style == 'mean-reverting':
            # For mean-reverting, flip the directions.
            if long_short == 'long-short':
                if rank_val > (1 - p):
                    return -1
                elif rank_val < p:
                    return 1
                else:
                    return 0
            elif long_short == 'long':
                return 1 if rank_val < p else 0
            elif long_short == 'short':
                return -1 if rank_val > (1 - p) else 0
            else:
                raise ValueError("Invalid long_short value. Must be one of 'long-short', 'long', or 'short'.")
        else:
            raise ValueError("Invalid style. Must be either 'momentum' or 'mean-reverting'.")
    
    # Apply the signal generation to each computed percentile rank.
    signals = percentiles_series.apply(generate_signal)
    
    # Align the signals with the full date index of price_df. Here we reindex, forward-fill (assuming signal persists)
    # and fill any remaining missing values with 0.
    signals_full = pd.Series(index=price_df.index, dtype=float)
    signals_full.loc[signals.index] = signals
    signals_full = signals_full.ffill().fillna(0)
    
    # Return a single-column DataFrame of signals.
    return pd.DataFrame({'signal': signals_full})

def rank_param(df, ascending=True):
    """
    Convert a wide-format parameter DataFrame (e.g., r2, pval, or beta) with index as j and columns as k 
    into a long-format DataFrame with columns: 'j', 'k', 'value', and 'rank'. 
    The ranking is done on the 'value' column in the order specified by `ascending`.

    Parameters:
        df (pd.DataFrame): DataFrame with index as j and columns as k.
        ascending (bool): If True, the lowest value gets rank 1, otherwise the highest gets rank 1.

    Returns:
        pd.DataFrame: A long-format DataFrame with columns ['j', 'k', 'value', 'rank'].
    """
    # Convert from wide to long format.
    long_df = df.stack().reset_index()
    long_df.columns = ['j', 'k', 'value']
    
    # Compute rank based on 'value'.
    long_df['rank'] = long_df['value'].rank(method='min', ascending=ascending)
    
    # Optionally, sort by rank.
    long_df = long_df.sort_values('rank').reset_index(drop=True)
    
    return long_df

def get_signals(price_df, weights, j, p, long_short='long-short', style='momentum', min_periods=None, window=None):
    """
    Computes trading signals based on the j-day previous log return percentile ranks.
    
    When weights is provided, the portfolio percentile rank is computed as the weighted sum 
    of individual asset returns. Otherwise, the function expects a single-ticker price DataFrame.
    
    Parameters:
      price_df : pd.DataFrame
          Prices of assets with dates as index and tickers as columns.
      weights : pd.DataFrame or None
          Weights for each ticker (same shape as price_df) or None (if only one ticker is present).
      j : int
          Look-back period for computing j-day log returns.
      p : float
          Threshold percentile (between 0 and 0.5) for generating signals.
      long_short : str, optional
          One of 'long-short', 'long', or 'short'. Default is 'long-short'.
      style : str, optional
          Either 'momentum' or 'mean-reverting'. Default is 'momentum'.
      min_periods : int, optional
          Minimum number of observations required for computing percentile rank.
      window : int, optional
          Window size to use when computing the historical percentile rank.
          
    Returns:
      pd.DataFrame
          A single-column DataFrame with the trading signal for each date.
    """
    # Validate dimensions.
    if weights is None:
        if price_df.shape[1] != 1:
            raise ValueError("Weights must be provided when multiple tickers are present.")
    else:
        if weights.shape != price_df.shape:
            raise ValueError("Weights must have the same shape as price_df.")
    
    # Compute the portfolio's j-day previous percentile rank using the helper.
    # When weights is provided, the helper computes a weighted portfolio; otherwise, it expects a single ticker.
    prev_df = add_prev_return_percentile_ranks(price_df, weights, j, window=window, min_periods=min_periods)
    percentile_series = prev_df['prev_p']
    
    # Define the signal generation function.
    def generate_signal(rank_val):
        # Treat missing values as neutral signal.
        if rank_val is None or (isinstance(rank_val, float) and np.isnan(rank_val)):
            return 0
        if style == 'momentum':
            if long_short == 'long-short':
                if rank_val > (1 - p):
                    return 1
                elif rank_val < p:
                    return -1
                else:
                    return 0
            elif long_short == 'long':
                return 1 if rank_val > (1 - p) else 0
            elif long_short == 'short':
                return -1 if rank_val < p else 0
            else:
                raise ValueError("Invalid long_short value. Must be one of 'long-short', 'long', or 'short'.")
        elif style == 'mean-reverting':
            # For mean-reverting strategies, flip the signal direction.
            if long_short == 'long-short':
                if rank_val > (1 - p):
                    return -1
                elif rank_val < p:
                    return 1
                else:
                    return 0
            elif long_short == 'long':
                return 1 if rank_val < p else 0
            elif long_short == 'short':
                return -1 if rank_val > (1 - p) else 0
            else:
                raise ValueError("Invalid long_short value. Must be one of 'long-short', 'long', or 'short'.")
        else:
            raise ValueError("Invalid style. Must be either 'momentum' or 'mean-reverting'.")
    
    # Generate signals based on the computed percentile ranks.
    signals = percentile_series.apply(generate_signal)
    
    # Align signals with the full date index of price_df by reindexing and filling any remaining missing values with 0.
    signals_full = pd.Series(index=price_df.index, dtype=float)
    signals_full.loc[signals.index] = signals
    signals_full = signals_full.fillna(0)
    
    # Return the signal as a single-column DataFrame.
    return pd.DataFrame({'signal': signals_full})