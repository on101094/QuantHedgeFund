"""
QS Research - Universe Screener

Filters the investment universe based on liquidity, volatility, and other criteria.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from loguru import logger


def universe_screener(
    df: pd.DataFrame,
    lookback_days: int = 730,
    volume_top_n: Optional[int] = 500,
    momentum_top_n: Optional[int] = None,
    percent_change_filter: bool = False,
    max_percent_change: float = 0.35,
    volatility_filter: bool = True,
    max_volatility: float = 0.25,
    min_avg_volume: int = 100_000,
    min_avg_price: float = 4.0,
    min_last_price: float = 5.0,
    symbol_column: str = "symbol",
    date_column: str = "date",
    close_column: str = "close",
    volume_column: str = "volume",
) -> pd.DataFrame:
    """
    Screen the investment universe based on multiple criteria.
    
    This screener helps avoid:
    - Illiquid stocks that are hard to trade
    - Penny stocks with high manipulation risk
    - Highly volatile stocks that add noise
    - Stocks with suspicious price movements
    
    Args:
        df: Price DataFrame
        lookback_days: Period for calculating metrics
        volume_top_n: Keep only top N by average volume
        momentum_top_n: Keep only top N by momentum
        percent_change_filter: Filter extreme daily changes
        max_percent_change: Maximum allowed daily change
        volatility_filter: Filter high volatility stocks
        max_volatility: Maximum allowed annualized volatility
        min_avg_volume: Minimum average daily volume
        min_avg_price: Minimum average price
        min_last_price: Minimum last traded price
        symbol_column: Name of symbol column
        date_column: Name of date column
        close_column: Name of close price column
        volume_column: Name of volume column
        
    Returns:
        Filtered DataFrame
    """
    initial_symbols = df[symbol_column].nunique()
    logger.info(f"Screening universe from {initial_symbols} symbols")
    
    df = df.copy()
    
    # Screening metrics use the latest lookback_days of data; the full history
    # of the symbols that pass is returned (backtests need it for factor warm-up)
    window = df
    if lookback_days:
        max_date = df[date_column].max()
        min_date = max_date - pd.Timedelta(days=lookback_days * 1.5)  # Buffer for weekends
        window = df[df[date_column] >= min_date]

    # Calculate screening metrics per symbol
    metrics = window.groupby(symbol_column).agg({
        close_column: ["mean", "std", "last"],
        volume_column: "mean",
    })
    metrics.columns = ["avg_price", "price_std", "last_price", "avg_volume"]

    # Calculate annualized volatility
    volatility = window.groupby(symbol_column).apply(
        lambda x: x[close_column].pct_change().std() * np.sqrt(252)
    )
    metrics["volatility"] = volatility
    
    # Apply filters
    valid_symbols = metrics.index.tolist()
    
    # Minimum average volume
    if min_avg_volume:
        vol_filter = metrics["avg_volume"] >= min_avg_volume
        filtered_out = (~vol_filter).sum()
        valid_symbols = metrics[vol_filter].index.tolist()
        logger.debug(f"Volume filter removed {filtered_out} symbols")
    
    # Minimum average price
    if min_avg_price:
        price_filter = metrics.loc[valid_symbols, "avg_price"] >= min_avg_price
        filtered_out = (~price_filter).sum()
        valid_symbols = [s for s, v in price_filter.items() if v]
        logger.debug(f"Avg price filter removed {filtered_out} symbols")
    
    # Minimum last price
    if min_last_price:
        last_filter = metrics.loc[valid_symbols, "last_price"] >= min_last_price
        filtered_out = (~last_filter).sum()
        valid_symbols = [s for s, v in last_filter.items() if v]
        logger.debug(f"Last price filter removed {filtered_out} symbols")
    
    # Volatility filter
    if volatility_filter:
        vol_filter = metrics.loc[valid_symbols, "volatility"] <= max_volatility
        filtered_out = (~vol_filter).sum()
        valid_symbols = [s for s, v in vol_filter.items() if v]
        logger.debug(f"Volatility filter removed {filtered_out} symbols")
    
    # Top N by volume
    if volume_top_n and len(valid_symbols) > volume_top_n:
        top_by_volume = (
            metrics.loc[valid_symbols]
            .nlargest(volume_top_n, "avg_volume")
            .index.tolist()
        )
        valid_symbols = top_by_volume
        logger.debug(f"Volume top_n reduced to {volume_top_n} symbols")
    
    # Filter DataFrame
    df = df[df[symbol_column].isin(valid_symbols)]
    
    final_symbols = df[symbol_column].nunique()
    logger.info(
        f"Screened universe: {final_symbols} symbols "
        f"({final_symbols/initial_symbols:.1%} retained)"
    )
    
    return df
