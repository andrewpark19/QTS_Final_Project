import polars as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_percentile_rank_series(series, interval: int = None, window: int = None, min_periods=None):
    """
    Compute the percentile rank for a numeric series.
    
    For each index i, the percentile rank is computed using a subset of historical returns:
    
    - If `interval` is provided as an integer, we select values at indices:
          i, i-interval, i-2*interval, ...
      If `interval` is None, then we use every index.
      
    - If `window` is None (default), the lookback is expanding (i.e. all available valid observations are used).
      If `window` is an integer, only the most recent `window` observations (based on the sampling determined by interval) are used.
      
    The percentile rank at index i is defined as:
    
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

# Example integration into the DataFrame processing:
def add_prev_return_percentile_ranks(df: pl.DataFrame, j: int, 
                                interval: int = None, window: int = None, 
                                min_periods: int = None) -> pl.DataFrame:
    """
    For a Polars DataFrame 'df' with a date column and asset price columns,
    compute the j-day log return for each asset, then calculate the percentile rank
    of that return using a historical lookback.
    
    Parameters:
      df: Polars DataFrame. The first column must be 'date', and the rest are price columns.
      j: lookback period in days to compute the log return:
             r_{t-j, t} = ln(P_t / P_{t-j})
      interval: if provided as an integer, only every interval-spaced return is used for the percentile rank.
                If None, every day's return is used.
                (Default behavior: use the passed j as the interval if desired, e.g. interval=j)
      window: if None (default), then the historical lookback is expanding (all available interval-spaced returns).
              If an integer, then a rolling window of that many observations is used.
      min_periods: If provided, requires at least min_periods valid returns in the selected window to compute the rank.
    
    Returns:
      A new Polars DataFrame with the 'date' column and, for each ticker, a column containing the percentile
      rank of the j-day log return computed over the selected historical window.
    """
    # Start with the date column.
    result = df.select("date")
    # Assume all columns except "date" are tickers.
    tickers = [col for col in df.columns if col != "date"]
    
    for ticker in tickers:
        ret_col = f"{ticker}_ret"
        # Compute the j-day log return. This will produce NaN for the first j rows.
        df = df.with_columns(
            (pl.col(ticker) / pl.col(ticker).shift(j)).log().alias(ret_col)
        )
        # Convert the return column to a list.
        ret_values = df[ret_col].to_list()
        
        # If interval is not provided, we default it to j (i.e. use every j-th day)
        eff_interval = j if interval is None else interval
        # Compute the percentile rank with the given parameters.
        percentiles = compute_percentile_rank_series(ret_values, interval=eff_interval, window=window, min_periods=min_periods)
        
        # Add the resulting percentile ranks to the output DataFrame.
        result = result.with_columns(pl.Series(name=ticker, values=percentiles))
    
    return result


def add_lookahead_return_percentile_ranks(df: pl.DataFrame, k: int, 
                                          interval: int = None, window: int = None, 
                                          min_periods: int = None) -> pl.DataFrame:
    """
    For a Polars DataFrame 'df' with a date column and asset price columns,
    compute the k-day look-ahead log return for each asset, then calculate the percentile rank
    of that look-ahead return over a historical window.
    
    The look-ahead log return is defined as:
    
         r_{t,t+k} = ln(P_{t+k} / P_t)
    
    where P_t is the asset price at time t.
    
    Parameters:
      df: Polars DataFrame. The first column must be 'date', and the remaining columns are asset prices.
      k: Holding period (in days) for the look-ahead return.
      interval: If provided as an integer, only every interval-spaced return is used for the percentile rank.
                If None, every day's look-ahead return is used.
                (Default behavior: if left as None, you can default to using k as the interval.)
      window: If None (default), then the lookback is expanding (i.e. all available valid observations are used).
              If an integer, then a rolling window of that many (interval-spaced) observations is used.
      min_periods: If provided, the percentile rank is computed only if there are at least this many valid returns in the selected window.
    
    Returns:
      A new Polars DataFrame with the 'date' column and, for each ticker, a column containing the percentile rank
      of the k-day look-ahead log return computed over the selected historical window.
    """
    # Start with the date column.
    result = df.select("date")
    # All columns except "date" are assumed to be asset tickers.
    tickers = [col for col in df.columns if col != "date"]
    
    for ticker in tickers:
        ret_col = f"{ticker}_lookahead_ret"
        # Compute the look-ahead log return: ln(P[t+k] / P[t]).
        # This will produce NaN for the last k rows.
        df = df.with_columns(
            (pl.col(ticker).shift(-k) / pl.col(ticker)).log().alias(ret_col)
        )
        # Convert the computed return column to a list.
        ret_values = df[ret_col].to_list()
        
        # If interval is not provided, default to k.
        effective_interval = k if interval is None else interval
        
        # Compute the percentile rank series.
        percentiles = compute_percentile_rank_series(ret_values, interval=effective_interval, window=window, min_periods=min_periods)
        
        # Add the computed look-ahead percentile ranks to the result DataFrame.
        result = result.with_columns(pl.Series(name=ticker, values=percentiles))
    
    return result

def equal_weight_portfolio(df: pl.DataFrame):
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
    
    return portfolio_df, weights_df

def marketcap_weighted_portfolio(return_df: pl.DataFrame, mcap_df: pl.DataFrame):
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
    
    return portfolio_df, weights_df


def compute_regression_matrices(rets: pl.DataFrame, j_list: list, k_list: list):
    """
    Given a Polars DataFrame 'rets' with a "date" column and a single ticker column,
    and lists of lookback parameters j_list and holding period parameters k_list,
    this function:
    
      1. Checks that 'rets' has only one ticker column (besides "date").
      2. For each combination of j (row: previous-return lookback) and k (column: look-ahead holding period):
            - Computes the previous return percentile series: p_{t-j,t}
              using add_prev_return_percentile_ranks(rets, j)
            - Computes the look-ahead return percentile series: p_{t,t+k}
              using add_lookahead_return_percentile_ranks(rets, k)
            - Merges the two series on "date", drops rows with NA values in either series,
              and then runs a linear regression:
                look_ahead_p = alpha + beta * prev_p + epsilon
            - Saves the beta coefficient and its p-value.
      3. Returns two pandas DataFrames:
            - beta_df: rows indexed by j values, columns labeled by k values.
            - pval_df: same structure containing the corresponding p-values.
            
    Parameters:
      rets: Polars DataFrame with columns ["date", ticker]
      j_list: list of integer j parameters (for previous return lookback). These will index the rows.
      k_list: list of integer k parameters (for look-ahead holding period). These will label the columns.
      
    Returns:
      beta_df, pval_df
    """
    # Check that there is only one ticker column (excluding "date")
    ticker_cols = [col for col in rets.columns if col != "date"]
    if len(ticker_cols) != 1:
        raise ValueError("Input DataFrame must have exactly one ticker column (excluding 'date').")
    ticker = ticker_cols[0]
    
    # Initialize dictionaries to collect regression results.
    # Outer keys will be j (rows) and inner keys will be k (columns).
    beta_results = {}
    pval_results = {}
    
    # Outer loop: iterate over each j (previous-return lookback) with a progress bar.
    for j in tqdm(j_list, desc="Processing j parameters (rows)"):
        beta_results[j] = {}
        pval_results[j] = {}
        
        # Compute the previous return percentile series for parameter j.
        prev_df = add_prev_return_percentile_ranks(rets, j)
        # Rename the ticker column to "prev" for clarity.
        prev_df = prev_df.rename({ticker: "prev"})
        
        # Inner loop: iterate over each k (look-ahead holding period) with a nested progress bar.
        for k in tqdm(k_list, desc=f"j={j}: Processing k parameters (columns)", leave=False):
            # Compute the look-ahead return percentile series for parameter k.
            look_df = add_lookahead_return_percentile_ranks(rets, k)
            # Rename the ticker column to "look" for clarity.
            look_df = look_df.rename({ticker: "look"})
            
            # Merge the two DataFrames on "date" and drop rows with NA in either series.
            merged = prev_df.join(look_df, on="date", how="inner")
            merged = merged.drop_nulls(subset=["prev", "look"])
            
            # If too few observations remain, store NaN.
            if merged.height < 10:
                beta_results[j][k] = np.nan
                pval_results[j][k] = np.nan
                continue
                
            # Convert the merged Polars DataFrame to a pandas DataFrame for statsmodels.
            mdf = merged.to_pandas()
            
            # Set up the regression: y (look) = alpha + beta * prev + error.
            X = sm.add_constant(mdf["prev"])
            y = mdf["look"]
            model = sm.OLS(y, X).fit()
            beta_results[j][k] = model.params["prev"]
            pval_results[j][k] = model.pvalues["prev"]
    
    # Convert nested dictionaries to pandas DataFrames.
    # Rows correspond to j (previous lookback), and columns correspond to k (look-ahead holding period).
    beta_df = pd.DataFrame(beta_results).T.sort_index()
    pval_df = pd.DataFrame(pval_results).T.sort_index()
    
    # Rename index and columns explicitly
    beta_df.index.name = "j"
    beta_df.columns.name = "k"
    
    pval_df.index.name = "j"
    pval_df.columns.name = "k"
    
    return beta_df, pval_df

def plot_heatmaps_j_k(beta_df: pd.DataFrame, pval_df: pd.DataFrame):
    """
    Given two DataFrames:
       - beta_df: rows indexed by j values and columns labeled by k values.
       - pval_df: same structure containing p-values.
    This function creates two separate heatmaps (using matplotlib):
       one for beta coefficients and one for p-values.
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

def rank_param(df):
  # Melt the DataFrame to long format
  long_df = df.reset_index().melt(id_vars='j', var_name='k', value_name='value')
  # Rename the columns for clarity
  long_df.columns = ['j', 'k', 'value']
  # Rank the values from min to max
  long_df['rank'] = long_df['value'].rank()
  # Sort the DataFrame by rank
  long_df = long_df.sort_values(by='rank')
  return long_df