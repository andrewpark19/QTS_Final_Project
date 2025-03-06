import polars as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

# -------------------------------
# Helper Function: Percentile Rank
# -------------------------------
def compute_percentile_rank_series(series, interval: int = 1, window: int = None, min_periods=None):
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
        if interval == 1:
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
def add_prev_return_percentile_ranks(prices, weights, j, window=None, min_periods=None, interval=None):
    """
    Computes the percentile ranks for the j-day previous log return.
    
    If weights is provided, it computes the portfolio return as a weighted sum of individual asset returns.
    If weights is None, prices is assumed to contain only one ticker.
    Returns a DataFrame with a single column 'prev_p' and the same index as prices.
    """
    # Compute j-day log returns: r_{t-j,t} = ln(P_t / P_{t-j})
    log_returns = np.log(prices / prices.shift(j))
    
    if not interval:
        interval=j

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
    percentiles = compute_percentile_rank_series(ret_list, interval=interval, window=window, min_periods=min_periods)
    return pd.DataFrame({'prev_p': percentiles}, index=prices.index)

def add_lookahead_return_percentile_ranks(prices, weights, k, window=None, min_periods=None, interval=None):
    """
    Computes the percentile ranks for the k-day look-ahead log return.
    
    If weights is provided, it computes the portfolio look-ahead return as a weighted sum.
    If weights is None, prices is assumed to contain only one ticker.
    Returns a DataFrame with a single column 'lookahead_p' and the same index as prices.
    """
    # Compute k-day lookahead log returns: r_{t,t+k} = ln(P_{t+k} / P_t)
    lookahead_returns = np.log(prices.shift(-k) / prices)
    
    if not interval:
        interval=k

    if weights is not None:
        if weights.shape != prices.shape:
            raise ValueError("Weights must have the same shape as prices.")
        port_return = (lookahead_returns * weights).sum(axis=1)
    else:
        if prices.shape[1] != 1:
            raise ValueError("When weights is None, prices should have a single ticker.")
        port_return = lookahead_returns.iloc[:, 0]
    
    ret_list = port_return.tolist()
    percentiles = compute_percentile_rank_series(ret_list, interval=interval, window=window, min_periods=min_periods)
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
def plot_heatmaps(beta_df: pd.DataFrame, pval_df: pd.DataFrame, r2_df: pd.DataFrame, sig_lvl: float = 0.05):
    """
    Given three DataFrames:
       - beta_df: rows indexed by j values and columns labeled by k values.
       - pval_df: same structure containing p-values.
       - r2_df: same structure containing R² values.
    This function creates three separate heatmaps (using matplotlib):
       one for beta coefficients, one for p-values, and one for R² values.
    Additionally, it creates another p-value heatmap that masks any values above the significance 
    level (sig_lvl), showing only values between 0 and sig_lvl.
    """
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
    
    # Plot masked P-Value heatmap (only show values <= sig_lvl).
    plt.figure(figsize=(8, 6))
    # Create a masked array: mask values greater than sig_lvl.
    masked_pvals = np.ma.masked_where(pval_df.values > sig_lvl, pval_df.values)
    im = plt.imshow(masked_pvals, aspect='auto', cmap='viridis', vmin=0, vmax=sig_lvl)
    plt.colorbar(im, label=f'P-Value (0 to {sig_lvl})')
    plt.xticks(ticks=np.arange(len(pval_df.columns)), labels=pval_df.columns)
    plt.yticks(ticks=np.arange(len(pval_df.index)), labels=pval_df.index)
    plt.title("Masked P-Value Heatmap (values > sig_lvl blanked out)")
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
            (pl.col(f"{coin}_mcap") / pl.col("total_mcap")).alias(f"w_{coin}")
        )
    
    # Compute the market-cap weighted portfolio return.
    # For each coin, multiply its return (from the return_df) by its weight and sum.
    portfolio_expr = sum(pl.col(f"w_{coin}") * pl.col(coin) for coin in common_coins)
    portfolio_df = daily.with_columns(
        portfolio_expr.alias("MCW_return")
    ).select(["date", "MCW_return"])
    
    # Create a weights DataFrame with date and each coin's weight.
    weight_cols = [f"w_{coin}" for coin in common_coins]
    weights_df = daily.select(["date"] + weight_cols)
    weights_df = weights_df.rename({col: col.replace('w_', '') for col in weights_df.columns})
    
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
            return np.nan
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
    
    # # Align signals with the full date index of price_df by reindexing 
    # signals_full.loc[signals.index] = signals
    # signals_full = signals_full.fillna(0)
    
    # # Return the signal as a single-column DataFrame.
    # return pd.DataFrame({'signal': signals_full})
    
    return pd.DataFrame(signals.rename('signal'))

def run_backtest(
    price_data: pd.DataFrame,
    weights: pd.DataFrame,
    signals: pd.DataFrame,
    k: int,
    tx_cost: float = 0.0015,
    start_day=None
):
    """
    Run a backtest given price data, weights, signals, a holding period (k), and transaction cost.
    
    Steps:
      1. Perform initial checks:
         - If weights is None, price_data must be a single column.
         - price_data, weights, and signals must have the same index length.
         - price_data and weights must be aligned in columns.
         - tx_cost must be between 0 and 1.
         
      2. Determine the first rebalancing date based on start_day:
         - If start_day is None, take the first non-NA date from signals.
         - Otherwise, take the first date that is on or after the first non-NA date that falls on the desired start_day (e.g., 'Mon').
         
      3. Sample rebalancing dates using the first_date and every k-th date thereafter.
      
      4. Filter the signals and weights DataFrames to only the rebalancing dates.
      
      5. Create holding weights based on signals:
         - If signal == 1, hold the given weights.
         - If signal == -1, hold the negative of the given weights.
         - If signal == 0, hold zero weights.
         
      6. Compute daily returns from the price_data (first row will be NA).
      
      7. Upsample the holding weights to daily frequency by forward-filling and then shift by 1 day to represent positions held.
      
      8. Compute the daily mark-to-market PnL as the elementwise product of the daily positions and daily returns,
         summing across assets to obtain the portfolio return.
      
      9. Compute transaction cost PnL:
         - Prepend an initial row of zeros to the holding weights.
         - Compute the day-to-day change in weights, take the absolute value, sum across assets, and multiply by tx_cost.
         - Transaction cost PnL is the negative of this cost.
      
      10. Form a results DataFrame with two columns (daily_mtm_pnl and tcost_pnl), compute total_pnl and cumulative_pnl.
      
      11. Concatenate the signal column and create a positions column based on the cumulative sum of signals (shifted by 1).
      
      12. Return a final DataFrame with index matching price_data and columns:
           ['signal', 'position', 'daily_mtm_pnl', 'tcost_pnl', 'total_pnl', 'cumulative_pnl'].
    """
    # ------------------------------
    # Initial Checks
    # ------------------------------
    if weights is None:
        if price_data.shape[1] != 1:
            raise ValueError("Weights must be provided when multiple tickers are present.")
    else:
        if weights.shape != price_data.shape:
            raise ValueError("Weights must have the same shape as price_data.")
    
    if not (len(price_data) == len(signals)):
        raise ValueError("price_data and signals must be aligned in length.")
    
    if not (tx_cost >= 0 and tx_cost < 1):
        raise ValueError("Transaction cost must be between 0 and 1.")
    
    # Ensure the index is a DatetimeIndex.
    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise ValueError("price_data index must be a pandas DatetimeIndex.")
    
    # ------------------------------
    # 1. Determine first rebalancing date
    # ------------------------------
    # Find the first non-NA date from signals.
    first_valid_date = signals.dropna().index[0]
    
    if start_day is None:
        first_date = first_valid_date
    else:
        # start_day can be a string like 'Mon', 'Tue', etc.
        # Find the first date on/after first_valid_date with day name matching start_day.
        # Normalize to lower-case for matching.
        target = start_day.lower()
        dates_after = price_data.loc[first_valid_date:].index
        first_date = None
        for d in dates_after:
            if d.day_name().lower().startswith(target):
                first_date = d
                break
        if first_date is None:
            raise ValueError(f"No date matching start_day '{start_day}' found on/after {first_valid_date}.")
    
    # ------------------------------
    # 2. Sample rebalancing dates based on first_date and k
    # ------------------------------
    # Use only dates on or after first_date from the price_data index.
    valid_dates = price_data.loc[first_date:].index
    # Take every kth date from valid_dates.
    rebal_dates = valid_dates[::k]
    
    # ------------------------------
    # 3 & 4. Filter signals and weights to rebalancing dates.
    # ------------------------------
    signals_rebal = signals.loc[rebal_dates]
    if weights is not None:
        weights_rebal = weights.loc[rebal_dates]
    else:
        weights_rebal = None  # Single ticker case.
    
    # ------------------------------
    # 5. Produce holding weights based on signals.
    # ------------------------------
    # For each rebalancing date, multiply the weights by the signal.
    # For multi-ticker: holding weight = weight * signal (broadcasted).
    # For single ticker: it's just signal if signal != 0, otherwise 0.
    if weights_rebal is not None:
        # Assume signals_rebal is a DataFrame with one column 'signal'
        # Multiply each row in weights_rebal by the corresponding signal.
        # If signal is 1, retain weight; if -1, reverse sign; if 0, becomes 0.
        holding_weights_rebal = weights_rebal.multiply(signals_rebal['signal'], axis=0)
    else:
        # Single ticker case.
        holding_weights_rebal = signals_rebal.copy()  # single column, where value is 1, -1, or 0.
    
    # ------------------------------
    # 6. Get the daily returns on close price.
    # ------------------------------
    # We use simple returns: (P_t / P_{t-1} - 1)
    daily_returns = price_data.pct_change()
    
    # ------------------------------
    # 7. Upsample the holding weights to daily frequency.
    # ------------------------------
    # Reindex holding_weights_rebal to the full price_data index, forward fill and fill NAs with 0.
    holding_weights_daily = holding_weights_rebal.reindex(price_data.index, method='ffill').fillna(0)
    # Shift by one day so that the weights we hold on day t were determined at the previous rebalancing.
    positions = holding_weights_daily.shift(1)
    
    # ------------------------------
    # 8. Compute daily mark-to-market PnL.
    # ------------------------------
    # Elementwise product of positions and daily returns.
    # For multi-ticker, sum across columns.
    if positions.ndim == 2 and positions.shape[1] > 1:
        daily_mtm_pnl = (positions * daily_returns).sum(axis=1)
    else:
        daily_mtm_pnl = positions.squeeze() * daily_returns.squeeze()
    
    # ------------------------------
    # 9. Compute transaction cost PnL.
    # ------------------------------
    # Insert an initial row of zeros to the holding weights for computing differences.
    # We assume holding_weights_daily represents the weights after rebalancing.
    initial = pd.DataFrame(0, index=[price_data.index[0]], columns=holding_weights_daily.columns)
    full_weights = pd.concat([initial, holding_weights_daily])
    # Compute the absolute difference day-to-day, sum across assets, then multiply by tx_cost.
    weight_changes = full_weights.diff().abs().iloc[1:]  # first diff will be NaN
    tcost = weight_changes.sum(axis=1) * tx_cost
    tcost_pnl = -tcost  # costs are negative.
    # Align transaction cost PnL with price_data index.
    tcost_pnl = tcost_pnl.reindex(price_data.index).fillna(0)
    
    # ------------------------------
    # 10. Create final PnL DataFrame.
    # ------------------------------
    pnl_df = pd.DataFrame({
        'daily_mtm_pnl': daily_mtm_pnl,
        'tcost_pnl': tcost_pnl
    })
    pnl_df['total_pnl'] = pnl_df['daily_mtm_pnl'] + pnl_df['tcost_pnl']
    # Compute cumulative pnl: starting with 1 and compounding daily returns.
    pnl_df['cumulative_pnl'] = (1 + pnl_df['total_pnl']).cumprod() - 1
    
    # ------------------------------
    # 11. Add signal and positions columns.
    # ------------------------------
    # For signals, reindex to full price_data index and fill missing values with 0.
    signals_full = signals['signal'].reindex(price_data.index).fillna(0)
    # Create a "positions" column from the signal column by taking the cumulative sum and shifting by 1.
    position_series = signals_full.cumsum().shift(1).fillna(0)
    
    # ------------------------------
    # 12. Final DataFrame assembly.
    # ------------------------------
    final_df = pnl_df.copy()
    final_df['signal'] = signals_full
    final_df['position'] = position_series
    # Order columns as specified.
    final_df = final_df[['signal', 'position', 'daily_mtm_pnl', 'tcost_pnl', 'total_pnl', 'cumulative_pnl']]
    
    return final_df

def buy_and_hold_cumulative(daily_returns: pd.Series) -> pd.Series:
    """
    Given a Series of daily returns (decimal fractions), 
    compute the buy-and-hold cumulative return over time.
    """
    daily_returns = daily_returns.fillna(0)
    return (1 + daily_returns).cumprod() - 1

def sweep_backtests(
    prices_df: pl.DataFrame,
    market_caps: pl.DataFrame,
    ret_df: pl.DataFrame,
    tcost: float = 0.0015,
    styles: list = ['momentum', 'mean-reverting'],
    strats: list = ['long', 'long-short', 'short'],
    weights_list: list = ['mcap', 'equal', 'BTC', 'ETH'],
    j_list: list = [7, 14, 21, 28, 35, 42],
    k_list: list = [3, 5, 7, 14, 21, 28, 35, 42],
    p_list: list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    windows: list = [None],
    burn_in_periods: list = [None]
):
    """
    Sweep over parameter combinations to run the backtest and save the final result DataFrames,
    with extra columns to record the parameter values used.

    The master parameter list (i.e. the parameter columns added to the result DataFrame) is built dynamically,
    including only those parameters whose input list has more than one element.

    NOTE:
      - For each j and k combination, only run the backtest if k >= j.
      - For the weights parameter:
           if weight == 'mcap': use marketcap_weighted_portfolio(ret_df, market_caps)
           if weight == 'equal': use equal_weight_portfolio(ret_df)
           else (e.g. 'BTC', 'ETH'): set weights to None (i.e. a single-ticker backtest).

    Returns:
      all_results_df: A concatenated DataFrame of all backtest results.
      params_df: A DataFrame listing the parameter combinations that were run.
    """
    
    # Prepare a list to collect individual result DataFrames and parameter dicts.
    results_list = []
    params_list = []
    
    # Determine which parameters are being swept (i.e. list length > 1)
    param_names = {}
    if len(styles) > 1:
        param_names['style'] = True
    if len(strats) > 1:
        param_names['strat'] = True
    if len(weights_list) > 1:
        param_names['weight'] = True
    if len(j_list) > 1:
        param_names['j'] = True
    if len(k_list) > 1:
        param_names['k'] = True
    if len(p_list) > 1:
        param_names['p'] = True
    if len(windows) > 1:
        param_names['window'] = True
    if len(burn_in_periods) > 1:
        param_names['min_period'] = True

    # Build the full parameter grid using product.
    full_grid = list(product(styles, strats, weights_list, j_list, k_list, p_list, windows, burn_in_periods))
    total_runs = len(full_grid)
    
    # Loop over parameter combinations with tqdm progress.
    for style, strat, weight_choice, j, k, p, window, min_period in tqdm(full_grid, total=total_runs, desc="Sweeping params"):
        # Only run if k >= j.
        if k < j:
            continue
        
        # 1. Determine the weight dataframe to use.
        if weight_choice == 'mcap':
            # Use market cap weights. The helper returns two objects; we discard the first.
            _, weight_df = marketcap_weighted_portfolio(ret_df, market_caps)
        elif weight_choice == 'equal':
            # Use equal weights.
            _, weight_df = equal_weight_portfolio(ret_df)
        else:
            # For individual tickers, pass weights as None.
            weight_df = None
        
        # 2. Change close price to pd df and ensure it is aligned with weights. 
        # Only take the relevant ticker column as well if doing individual analysis.
        close_prices = prices_df.to_pandas()
        close_prices.set_index('date', inplace=True)
        if not isinstance(weight_df, pd.DataFrame):
            close_prices = close_prices[[weight_choice]]
        else:
            close_prices, weight_df = close_prices.align(weight_df, join='inner', axis=0)
            
        
        # 3. Generate signals.
        signals = get_signals(
            price_df=close_prices,
            weights=weight_df,
            j=j,
            p=p,
            long_short=strat,
            style=style,
            min_periods=min_period,
            window=window
        )
        
        # 4. Run backtest.
        result_df = run_backtest(
            price_data=close_prices,
            weights=weight_df,
            signals=signals,
            k=k,
            tx_cost=tcost
        )
        
        # Add parameter columns to result_df.
        # Only include those parameters whose input list length > 1.
        param_dict = {}
        if 'style' in param_names:
            param_dict['style'] = style
        if 'strat' in param_names:
            param_dict['strat'] = strat
        if 'weight' in param_names:
            param_dict['weight'] = weight_choice
        if 'j' in param_names:
            param_dict['j'] = j
        if 'k' in param_names:
            param_dict['k'] = k
        if 'p' in param_names:
            param_dict['p'] = p
        if 'window' in param_names:
            param_dict['window'] = window
        if 'min_period' in param_names:
            param_dict['min_period'] = min_period
        
        # Add these parameter columns to every row of result_df.
        for key, val in param_dict.items():
            result_df[key] = val
        
        results_list.append(result_df)
        # Also store the parameter combination used for this run.
        params_list.append(param_dict)
    
    # Concatenate all backtest result DataFrames.
    if results_list:
        all_results_df = pd.concat(results_list)
    else:
        all_results_df = pd.DataFrame()
    
    # Convert params_list to a DataFrame (each row is a parameter combination used).
    params_df = pd.DataFrame(params_list)
    
    # Cleaning
    metrics_list = all_results_df.columns[:6].to_list()
    params_list = all_results_df.columns[6:].to_list()
    all_results_df = all_results_df[params_list + metrics_list]
    
    return all_results_df, params_df

def compute_run_metrics(run_df: pd.DataFrame, 
                        bench_SPY: pd.Series = None, 
                        bench_BTC: pd.Series = None):
    """
    Given a backtest result DataFrame (with a DatetimeIndex) that contains at least the following columns:
      - 'total_pnl': daily total portfolio return (in decimals)
      - 'cumulative_pnl': cumulative return (in decimals)
      - 'tcost_pnl': transaction cost PnL (used here to count trades)
      
    Compute the following metrics:
      1. total_return (%): final cumulative return * 100.
      2. ann_return (%): mean daily total return scaled by 252 * 100.
      3. ann_vol (%): daily total return volatility scaled by sqrt(252) * 100.
      4. ann_sharpe: ann_return / ann_vol (risk-free rate assumed 0).
      5. ann_sortino: ann_return divided by downside volatility (annualized).
      6. max_drawdown (%): maximum drop from a running peak of cumulative_pnl.
      7. calmar: ann_return divided by absolute max_drawdown.
      8. downside_beta_SPY: beta computed on days when SPY return < 0.
      9. downside_beta_BTC: similarly for BTC.
     10. var_95 (daily, %): 5th percentile of daily total return * 100.
     11. cvar_95 (daily, %): average of daily returns below the 5th percentile * 100.
     12. total_trades: count of days where absolute transaction cost > 0.
     
    If bench_SPY or bench_BTC are provided, they are assumed to be aligned (by date) with run_df 
    and represent the benchmark daily returns.
    
    Returns a dictionary of metrics.
    """
    # Ensure the dataframe is sorted by date.
    run_df = run_df.sort_index()
    # 1. Total return (%)
    try:
        final_cum = run_df['cumulative_pnl'].iloc[-1]
    except IndexError:
        final_cum = np.nan
    total_return = final_cum

    # 2. Annualized return (%)
    ann_return = run_df['total_pnl'].mean() * 252

    # 3. Annualized volatility (%)
    ann_vol = run_df['total_pnl'].std() * np.sqrt(252)

    # 4. Annualized Sharpe
    ann_sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    # 5. Annualized Sortino Ratio.
    # Downside volatility: standard deviation of returns when they are below 0.
    downside_std = run_df['total_pnl'][run_df['total_pnl'] < 0].std()
    ann_downside = downside_std * np.sqrt(252)
    ann_sortino = ann_return / ann_downside if ann_downside != 0 else np.nan

    # 6. Max Drawdown (%)
    equity = run_df['cumulative_pnl']+1
    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    max_drawdown = drawdown.min() * 100

    # 7. Calmar Ratio
    calmar = ann_return*100 / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # 8 & 9. Downside Beta for SPY and BTC.
    def compute_downside_beta(asset_ret, bench_ret):
        # Only consider days when the benchmark is negative.
        mask = bench_ret < 0
        if mask.sum() < 2:
            return np.nan
        cov = np.cov(asset_ret[mask], bench_ret[mask])[0, 1]
        var = np.var(bench_ret[mask])
        return cov / var if var != 0 else np.nan

    downside_beta_SPY = np.nan
    downside_beta_BTC = np.nan
    if bench_SPY is not None:
        downside_beta_SPY = compute_downside_beta(run_df['total_pnl'], bench_SPY)
    if bench_BTC is not None:
        downside_beta_BTC = compute_downside_beta(run_df['total_pnl'], bench_BTC)
    
    # 10. VaR (95%, daily, in %)
    var_95 = np.percentile(run_df['total_pnl'].dropna(), 5) * 100

    # 11. CVaR (95%, daily, in %)
    losses = run_df['total_pnl'][run_df['total_pnl'] <= np.percentile(run_df['total_pnl'].dropna(), 5)]
    cvar_95 = losses.mean() * 100 if len(losses) > 0 else np.nan

    # 12. Total (1-way) Trades: count days where abs(tcost_pnl) > 0.
    total_trades = (run_df['tcost_pnl'].abs() > 0).sum()
    
    metrics = {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'ann_sharpe': ann_sharpe,
        'ann_sortino': ann_sortino,
        'max_drawdown (%)': max_drawdown,
        'calmar': calmar,
        'downside_beta_SPY': downside_beta_SPY,
        'downside_beta_BTC': downside_beta_BTC,
        'var_95 (%)': var_95,
        'cvar_95 (%)': cvar_95,
        'total_trades': total_trades
    }
    return metrics

def analyze_backtests(all_results_df: pd.DataFrame, 
                      params_df: pd.DataFrame,
                      other_returns: pd.DataFrame):
    """
    For each parameter set (row) in params_df, filter all_results_df for that run,
    drop the parameter columns from the filtering DataFrame, and compute performance metrics.
    
    Then, also compute the metrics for each of the special benchmark time series (SPY, BTC, and Crypto Market)
    contained in other_returns. The date index of other_returns is aligned to the testing period
    (i.e. between the min and max dates of all_results_df).
    
    The function returns a DataFrame where each row corresponds to a run (or benchmark) with columns:
       - Parameter columns (only those that were swept)
       - The following metrics: ['total_return', 'ann_return', 'ann_vol', 'ann_sharpe', 'ann_sortino', 
         'max_drawdown', 'calmar', 'downside_beta_SPY', 'downside_beta_BTC', 'var_95', 'cvar_95', 'total_trades'].
         
    Additionally, a separate DataFrame containing the parameter combinations run (params_df) is also returned.
    """
    # First, determine the date range of the backtests.
    start_date = all_results_df.index.min()
    end_date = all_results_df.index.max()
    
    # Align other_returns to the testing period.
    # Reindex other_returns to the date range of all_results_df.
    other_returns = pd.DataFrame(index=all_results_df.index.unique().sort_values()).join(other_returns, how='left').fillna(0)
    
    # Determine which parameter columns were swept.
    possible_params = ['style', 'strat', 'weight', 'j', 'k', 'p', 'window', 'min_period']
    swept_params = [col for col in possible_params if col in params_df.columns and params_df[col].nunique() > 1]
    
    # Prepare a list to collect metric dictionaries.
    metrics_list = []
    
    # Iterate over each parameter run in params_df with tqdm progress bar.
    for idx, param_row in tqdm(params_df.iterrows(), total=len(params_df), desc="Processing backtests"):
        # Build a boolean mask for all_results_df for the current run.
        mask = pd.Series(True, index=all_results_df.index)
        for col in params_df.columns:
            # Only filter if the column exists in all_results_df.
            if col in all_results_df.columns:
                mask = mask & (all_results_df[col] == param_row[col])
        run_df = all_results_df[mask]
        # Drop the parameter columns for metric calculation.
        run_df = run_df.drop(columns=params_df.columns.intersection(all_results_df.columns), errors='ignore')
        # Ensure run_df is not empty.
        if run_df.empty:
            continue
        # Compute metrics for this run.
        metrics = compute_run_metrics(run_df, 
                                      bench_SPY=other_returns['SPY'], 
                                      bench_BTC=other_returns['BTC'])
        # Add the parameter values to the metrics dictionary.
        for col in swept_params:
            metrics[col] = param_row[col]
        # Also add all parameters from param_row (for consistency).
        for col in params_df.columns:
            if col not in metrics:
                metrics[col] = param_row[col]
        metrics_list.append(metrics)
    
    # Convert metrics list to DataFrame.
    metrics_df = pd.DataFrame(metrics_list)
    
    # Now compute metrics for the special benchmark series with tqdm progress.
    bench_results = []
    for bench in tqdm(['SPY', 'BTC', 'Crypto Market'], desc="Processing benchmarks"):
        # Create a mask for the benchmark: use the aligned other_returns for the testing period.
        bench_series = other_returns[bench]
        # Here we treat the benchmark return series as both the daily total pnl and use cumulative product for cumulative pnl.
        temp_df = pd.DataFrame(index=other_returns.index)
        temp_df['total_pnl'] = bench_series
        temp_df['cumulative_pnl'] = (1 + bench_series).cumprod()
        # For trades, set to 0.
        temp_df['tcost_pnl'] = 0
        bench_metrics = compute_run_metrics(temp_df, bench_SPY=None, bench_BTC=None)
        # For benchmark rows, fill parameter columns as NA except set 'strat' to the benchmark name.
        for col in possible_params:
            bench_metrics[col] = np.nan
        bench_metrics['strat'] = bench
        bench_results.append(bench_metrics)
    
    bench_df = pd.DataFrame(bench_results)
    
    # Concatenate run metrics and benchmark metrics.
    final_metrics_df = pd.concat([metrics_df, bench_df], ignore_index=True)
    
    # Cleaning
    metric_cols = final_metrics_df.columns[:12].to_list()
    param_cols = final_metrics_df.columns[12:].to_list()
    final_metrics_df = final_metrics_df[param_cols+metric_cols]
    final_metrics_df = final_metrics_df.dropna(axis=1, how='all')
    # Return final metrics DataFrame, dropping all na columns (no swept parameters)
    return final_metrics_df