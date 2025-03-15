import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional

def ts_mom_signals(
        price_data: pd.DataFrame,
        mcap_weights: pd.DataFrame,
        lookback_period: int,
        percentile_threshold: float,
        weighting_scheme: str,
        burn_in_period: str= None
) ->  pd.DataFrame:

    """
    price_data: DataFrame or array of prices indexed by date for each ticker (pivot table format)
    mcap_weights: DataFrame or array of weights based on market capitalizations indexed by date for each ticker (pivot table format)
    lookback_period: j, number of bars to look back (e.g., 20 for 20-day momentum)
    burn_in_period: i, number of bars to burn in (e.g., 20 for 20-day burn-in period) for expanding window
    percentile_threshold: e.g., 0.8 for top 20% threshold
    weighting_scheme: 'equal' for equal weights, 
                      'mcap' for market cap weights, 
                      '{ticker_name}' for individual tickers

    returns: DataFrame with columns:
             - 'signal': +1 for long, -1 for short, 0 for neutral
    """

    # Calculate lookback returns
    momentum_returns = np.log(price_data / price_data.shift(lookback_period))
    momentum_returns.dropna(inplace=True)

    # Portfolio Construction
    if weighting_scheme == 'equal':

        momentum_returns['Portfolio'] = momentum_returns.mean(axis=1)

    elif weighting_scheme == 'mcap':
        momentum_returns, mcap_weights = momentum_returns.align(mcap_weights, join='inner', axis=0)
        momentum_returns['Portfolio'] = (momentum_returns * mcap_weights).sum(axis=1) / mcap_weights.sum(axis=1)
    else:
        # Assuming weighting_scheme is a ticker name
        momentum_returns['Portfolio'] = momentum_returns[weighting_scheme]

    # Rank returns historically based off lookback period
    momentum_returns['rank'] = (
        momentum_returns['Portfolio']
        .expanding(min_periods = burn_in_period)
        .apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    )

    # Generate signals based on rank
    momentum_returns['signal'] = 0
    momentum_returns.loc[momentum_returns['rank'] > percentile_threshold, 'signal'] = 1
    momentum_returns.loc[momentum_returns['rank'] < 1-percentile_threshold, 'signal'] = -1


    # Also calculate daily returns for backtesting purposes
    daily_returns = price_data.pct_change()
    daily_returns = daily_returns.reindex(momentum_returns.index)

    if weighting_scheme == 'equal':
        daily_portfolio_return = daily_returns.mean(axis=1)
    elif weighting_scheme == 'mcap':
        daily_portfolio_return = (
            (daily_returns * mcap_weights).sum(axis=1) / mcap_weights.sum(axis=1)
        ).reindex(momentum_returns.index)
    else:
        daily_portfolio_return = daily_returns[weighting_scheme]

    # Ensure we fill any NaNs at the start
    daily_portfolio_return = daily_portfolio_return.fillna(0)

    momentum_returns['daily_portfolio_return'] = daily_portfolio_return


    return momentum_returns[['Portfolio', 'rank', 'signal', 'daily_portfolio_return']]



def backtest_strategy(
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        holding_period: int,
        tx_cost: float = 0.0015,
        strat: str = 'long_short',
        start_day: Optional[str] = None
) -> pd.DataFrame:

    """
    price_data: pd.DataFrame of prices indexed by date (pivoted: columns = {tickers or 'Portfolio'})
    signal_data: Output from ts_mom_signals() with columns ['Portfolio', 'rank', 'signal']
    holding_period: k, how many bars we hold each position before rebalancing
    style: 'momentum' or 'mean-reverting'
    strat: 'long-only', 'short-only', or 'long-short'
    tcost: transaction cost rate
    start_day: optional day of week (e.g., 'MON') to align rebalancing

    """
    
    df = signals[['signal', 'daily_portfolio_return']].copy()
    df.dropna(inplace=True)

    # Start days
    if start_day is not None:
        possible_starts = df.index[df.index.day_name() == start_day]
        if not possible_starts.empty:
            start_date = possible_starts[0]
            df = df[df.index >= start_date]
    
    # Strategy type
    if strat == 'long_only':
        df.loc[df['signal'] == -1, 'signal'] = 0
    elif strat == 'short_only':
        df.loc[df['signal'] == 1, 'signal'] = 0

    # Generate trade signals for each holding period
    rebal_dates = df.index[::holding_period]
    df['trade_signal'] = 0
    df.loc[rebal_dates, 'trade_signal'] = df['signal']


    # Track positions
    df['position'] = 0
    current_position = 0

    for i in range(1, len(df)):
        today_index = df.index[i]
        yday_index = df.index[i-1]

        if df.at[today_index, 'trade_signal'] != 0:
            current_position = df.at[today_index, 'trade_signal']
        else:
            current_position = df.at[yday_index, 'position']

        df.at[today_index, 'position'] = current_position

    # Calculate returns
    df['strategy_return'] = df['position'].shift(1) * df['daily_portfolio_return']
    df['strategy_return'].fillna(0, inplace=True)

    # Calculate transaction costs
    df['position_change'] = df['position'].diff().abs().fillna(0)
    df['tx_cost'] = df['position_change'] * tx_cost

    # Calculate net returns
    df['net_return'] = df['strategy_return'] - df['tx_cost']
    df['cumulative_return'] = (1 + df['net_return']).cumprod() - 1

    return df[['strategy_return', 'tx_cost', 'net_return', 'cumulative_return']]


def backtest_strategy_V2(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    holding_period: int,
    tx_cost: float = 0.0015,
    strat: str = 'mean_reverting',
    pos_type: str = 'long_short',
    start_day: Optional[str] = None,
    trades_as_df: bool = True
) -> Union[pd.DataFrame, dict]:
    """
    Backtest a time-series momentum or similar strategy on a single portfolio/asset, 
    using day-by-day mark-to-market logic and a trade log.

    Parameters
    ----------
    price_data : pd.DataFrame
        (Currently not used if 'signals' already contains daily returns. 
         Provided here for consistency or future expansion.)
    signals : pd.DataFrame
        Must contain:
          - 'signal': (+1, 0, or -1) for each day
          - 'daily_portfolio_return': the daily return (decimal fraction) for that asset/portfolio
    holding_period : int
        Rebalance frequency, e.g. every k days.
    tx_cost : float
        Transaction cost rate for flipping or opening/closing. 
        E.g., 0.0015 => 0.15% per 1.0 position change.
    strat : {'momentum', 'mean_reverting'}
        Strategy type for constraints or special logic.
    pos_type : {'long_short', 'long_only', 'short_only'}
        Constrain signals to only long, only short, or both.
    start_day : Optional[str]
        If provided (e.g., 'MON'), only start trading on the first date matching that day.
    trades_as_df : bool
        Whether to return the trade log as a DataFrame or a dict.

    Returns
    -------
    result : pd.DataFrame
        Contains columns for daily mark-to-market:
          - 'position': current position (+1, -1, or 0)
          - 'daily_mtm': daily open PnL from holding 'position' * daily_portfolio_return
          - 'open_pnl': running open PnL while position is open
          - 'trade_pnl': realized PnL upon closing/flipping
          - 'cumulative_return': sum of all net realized returns so far
          - 'daily_mtm_ret': same-day return as fraction of prior open position's scale
        Also attaches performance stats and a trade log under .attrs.

    or (result, stats_dict) if you prefer to return them separately.

    Notes
    -----
    - This version treats 'daily_portfolio_return' as the fraction gained/lost each day. 
      A position of +1 means you earn that fraction, -1 means you lose that fraction.
    - For a more robust approach with actual notional capital, 
      you would track 'cash' and 'total_equity' as in the spread strategy code.
    """

    # -----------------------------
    # 1. Prepare Data
    # -----------------------------
    df = signals[['signal', 'daily_portfolio_return']].copy()
    df.dropna(inplace=True)

    # (A) Optional: filter to start_day of the week
    if start_day is not None:
        possible_starts = df.index[df.index.day_name() == start_day]
        if not possible_starts.empty:
            start_date = possible_starts[0]
            df = df.loc[df.index >= start_date]

    # (B) Enforce strategy constraints
    if strat == 'mean_reverting':
        df['signal'] = -df['signal']

    
    if pos_type == 'long_only':
        df.loc[df['signal'] < 0, 'signal'] = 0
    elif pos_type == 'short_only':
        df.loc[df['signal'] > 0, 'signal'] = 0

    # (C) Determine rebalancing days
    first_signal_idx = df.index[df['signal'] != 0]
    df['trade_signal'] = 0
    if len(first_signal_idx) > 0:
        first_signal_date = first_signal_idx[0]
        first_signal_loc = df.index.get_loc(first_signal_date)
        
        # 2. Rebal from 'first_signal_loc' every holding_period
        rebal_locs = range(first_signal_loc, len(df), holding_period)
        rebal_dates = df.index[list(rebal_locs)]
        
        # 3. Assign trade_signal
        df.loc[rebal_dates, 'trade_signal'] = df['signal']
    else:
        print("No nonzero signals found. No rebalancing triggered.")

    # -----------------------------
    # 2. Initialize Columns
    # -----------------------------
    new_cols = [
        'position', 'daily_mtm', 'open_pnl', 'trade_pnl',
        'cumulative_return', 'daily_mtm_ret'
    ]
    for col in new_cols:
        df[col] = 0.0

    # We track the position in {+1, -1, 0} units
    position = 0.0
    open_pnl = 0.0           # Accumulated open PnL (unrealized)
    cum_return = 0.0         # Realized PnL in fractional terms
    trades_log = []

    # -----------------------------
    # 3. Main Loop
    # -----------------------------
    dates = df.index
    for i, today in enumerate(dates):
        if i == 0:
            first_signal = df.at[today, 'trade_signal']
            if first_signal != 0:
                cost_open = abs(first_signal) * tx_cost
                open_pnl -= cost_open  # immediate cost
                trades_log.append({
                    'date': today,
                    'action': 'OPEN',
                    'old_pos': 0,
                    'new_pos': first_signal,
                    'realized_pnl': float('nan'),
                    'tx_cost': cost_open
                })
                position = first_signal

            df.at[today, 'position'] = position
            df.at[today, 'cumulative_return'] = cum_return + open_pnl
            continue

        # (B) Daily mark-to-market from old position
        ret_t = df.at[today, 'daily_portfolio_return']
        daily_mtm = position * ret_t
        open_pnl += daily_mtm

        # (C) daily_mtm_ret for reporting
        daily_mtm_ret = daily_mtm if abs(position) > 0 else 0.0

        # (D) If it's a rebalance day => 'trade_signal' != 0
        reb_sig = df.at[today, 'trade_signal']
        if reb_sig != 0:
            old_pos = position
            # 1) Close old position if any
            if old_pos != 0:
                # Realize the open_pnl
                realized_pnl = open_pnl
                cost_close = abs(old_pos) * tx_cost
                realized_pnl -= cost_close
                cum_return += realized_pnl

                trades_log.append({
                    'date': today,
                    'action': 'CLOSE',
                    'old_pos': old_pos,
                    'new_pos': 0,
                    'realized_pnl': realized_pnl,
                    'tx_cost': cost_close
                })

                # Reset after close
                open_pnl = 0.0
                position = 0.0

            # 2) Open new position if reb_sig != 0
            if reb_sig != 0:
                cost_open = abs(reb_sig) * tx_cost
                open_pnl -= cost_open  # pay cost
                trades_log.append({
                    'date': today,
                    'action': 'OPEN' if old_pos == 0 else 'FLIP_OPEN',
                    'old_pos': 0,
                    'new_pos': reb_sig,
                    'realized_pnl': float('nan'),
                    'tx_cost': cost_open
                })
                position = reb_sig

        # (E) Store daily results
        df.at[today, 'position'] = position
        df.at[today, 'daily_mtm'] = daily_mtm
        df.at[today, 'daily_mtm_ret'] = daily_mtm_ret
        df.at[today, 'open_pnl'] = open_pnl

        # *** CRITICAL CHANGE: daily "cumulative_return" = realized PnL + open_pnl
        df.at[today, 'cumulative_return'] = cum_return + open_pnl

    # -----------------------------
    # 4. Force Close at End
    # -----------------------------
    final_date = df.index[-1]
    if position != 0:
        realized_pnl = open_pnl
        cost_close = abs(position) * tx_cost
        realized_pnl -= cost_close
        cum_return += realized_pnl

        trades_log.append({
            'date': final_date,
            'action': 'FORCED_CLOSE',
            'old_pos': position,
            'new_pos': 0,
            'realized_pnl': realized_pnl,
            'tx_cost': cost_close
        })

        # After final close, no open_pnl remains
        open_pnl = 0.0
        df.at[final_date, 'trade_pnl'] = realized_pnl

        # Update final row
        df.at[final_date, 'cumulative_return'] = cum_return + open_pnl
        df.at[final_date, 'position'] = 0
        df.at[final_date, 'open_pnl'] = 0.0

    # -----------------------------
    # 5. Basic Performance Stats
    # -----------------------------
    # daily_mtm_ret is the daily "PnL" in fraction form. You can compute Sharpe, etc.
    daily_returns = df['daily_mtm_ret'].fillna(0)
    if len(daily_returns) < 2:
        sharpe = 0.0
        sortino = 0.0
    else:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std(ddof=1)
        neg_ret = daily_returns[daily_returns < 0]
        dd_neg = neg_ret.std(ddof=1)

        sharpe = mean_ret / std_ret if std_ret > 1e-12 else 0.0
        sortino = mean_ret / dd_neg if dd_neg > 1e-12 else 0.0

    # We store stats and trades in .attrs for convenience
    performance_stats = {
        'final_cumulative_return': cum_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'num_trades': len(trades_log)
    }

    if trades_as_df and len(trades_log) > 0:
        trades_log = pd.DataFrame(trades_log).set_index('date')

    df.attrs['performance_stats'] = performance_stats
    df.attrs['trades_log'] = trades_log

    return df


def buy_and_hold_cumulative(daily_returns: pd.Series) -> pd.Series:
    """
    Given a Series of daily returns (decimal fractions), 
    compute the buy-and-hold cumulative return over time.
    """
    daily_returns = daily_returns.fillna(0)
    return (1 + daily_returns).cumprod() - 1

def show_performance(
    result_df: pd.DataFrame,
    title: str = "Strategy Performance",
    show_trades: bool = True,
    daily_ret_col: str = 'daily_portfolio_return'
):
    """
    Display a summary of the backtest performance,
    including a stats table, a single-plot cumulative returns chart
    (with optional buy & hold comparison), and optional trade markers.

    Parameters
    ----------
    result_df : pd.DataFrame
        The backtest results. Expected columns:
          - 'cumulative_return': strategy's running return
          - 'position': daily position (+1, -1, or 0) [optional if you want trade markers]
          - daily_ret_col (default 'daily_portfolio_return'):
                the underlying asset's daily returns for buy-and-hold comparison.
        Also expected:
          - result_df.attrs['performance_stats']: dict of stats
          - result_df.attrs['trades_log']: trade history (DataFrame or list of dicts).
    title : str
        Title for the plot.
    show_trades : bool
        If True, attempt to plot markers for trade entries/exits on the same axis.
    daily_ret_col : str
        Column name for the underlying asset's daily returns,
        used to compute a buy-and-hold cumulative return for comparison.
    """

    # 1. Retrieve the existing performance stats (strategy Sharpe, etc.)
    perf_stats = result_df.attrs.get('performance_stats', {})

    # 2. Compute Buy-and-Hold Sharpe Ratio if daily_ret_col is in the DataFrame
    if daily_ret_col in result_df.columns:
        # The buy-and-hold daily returns are simply the asset's daily returns
        bh_daily_returns = result_df[daily_ret_col].fillna(0)

        if len(bh_daily_returns) < 2:
            bh_sharpe = 0.0
        else:
            bh_mean = bh_daily_returns.mean()
            bh_std = bh_daily_returns.std(ddof=1)
            bh_sharpe = bh_mean / bh_std if bh_std > 1e-12 else 0.0

        # Add the BH Sharpe to the performance stats
        perf_stats['bh_sharpe'] = bh_sharpe

    # 3. Print or display the performance stats
    print("=== Performance Stats ===")
    if not perf_stats:
        print("No performance stats found in result_df.attrs['performance_stats'].")
    else:
        for key, val in perf_stats.items():
            print(f"{key:25s}: {val}")
    print()

    # 4. If trades_log is available, show the first few trades
    trades_log = result_df.attrs.get('trades_log', None)
    if trades_log is not None:
        print("=== Trades Log (head) ===")
        if isinstance(trades_log, pd.DataFrame):
            print(trades_log.head())
        else:
            # If it's a list of dicts, convert to DataFrame for display
            trades_df = pd.DataFrame(trades_log)
            trades_df.set_index('date', inplace=True, drop=False)
            print(trades_df.head())
    print()

    # 5. Create a single figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 6. Plot Strategy's Cumulative Return
    if 'cumulative_return' not in result_df.columns:
        print("Warning: 'cumulative_return' not found in columns.")
    else:
        ax.plot(
            result_df.index,
            result_df['cumulative_return'],
            label='Strategy Cumulative Return',
            color='blue'
        )

    # 7. Compute & Plot Buy-and-Hold if daily_ret_col is present
    if daily_ret_col in result_df.columns:
        bh_series = buy_and_hold_cumulative(result_df[daily_ret_col])
        ax.plot(
            result_df.index,
            bh_series,
            label='Buy & Hold',
            color='gray',
            linestyle='--',
            alpha=0.8
        )

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc='best')
    ax.grid(True)

    plt.tight_layout()
    plt.show()