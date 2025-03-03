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
        burn_in_period: Union[int, None] = None
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