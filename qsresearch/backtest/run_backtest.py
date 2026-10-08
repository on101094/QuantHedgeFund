"""
QS Research - Backtest Runner

Main backtesting execution engine with MLflow integration.
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import pickle

import pandas as pd
from loguru import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")

from config.settings import get_settings
from qsresearch.portfolio_analysis.performance_metrics import calculate_all_metrics


SIMULATOR_VERSION = "signal_weights_v2"
DEFAULT_REBALANCE_FREQUENCY = "M"
DEFAULT_TRANSACTION_COST_BPS = 5.0
REBALANCE_FREQUENCIES = {"D", "W", "M", "Q"}


class BacktestDataError(ValueError):
    """Raised when a backtest has no usable price or signal data."""


def run_backtest(
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Run a backtest with the given configuration.
    
    This is the main entry point for running backtests. It:
    1. Loads and preprocesses data
    2. Calculates factors/features
    3. Runs the backtesting engine
    4. Calculates performance metrics
    5. Logs results to MLflow
    
    Args:
        config: Backtest configuration dictionary containing:
            - bundle_name: Zipline bundle name
            - start_date: Backtest start date
            - end_date: Backtest end date
            - capital_base: Starting capital
            - preprocessing: List of preprocessing steps
            - algorithm: Algorithm function and params
            - portfolio_strategy: Portfolio construction config
            - simulation: Optional simulator settings
                - rebalance_frequency: 'D', 'W', 'M' (default) or 'Q'
                - transaction_cost_bps: cost per unit of turnover (default 5)
        output_dir: Directory to save results
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Dictionary with backtest results and metrics
    """
    settings = get_settings()
    
    if output_dir is None:
        output_dir = settings.dashboard_data_dir / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting backtest run")
    logger.info(f"Config: {config.get('experiment_name', 'unnamed')}")
    
    # Extract config values
    bundle_name = config.get("bundle_name", "historical_prices_fmp")
    start_date = config.get("start_date", "2015-01-01")
    end_date = config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
    capital_base = config.get("capital_base", 1_000_000)
    simulation_config = {
        "rebalance_frequency": DEFAULT_REBALANCE_FREQUENCY,
        "transaction_cost_bps": DEFAULT_TRANSACTION_COST_BPS,
        **config.get("simulation", {}),
    }
    
    # MLflow setup
    if log_to_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        experiment_name = config.get("experiment_name", settings.mlflow_experiment_name)
        mlflow.set_experiment(experiment_name)
        
        run_name = config.get("run_name", f"backtest_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
        
        # Use nested=True to support batch runs and parameter sweeps
        mlflow.start_run(run_name=run_name, nested=True)
        
        # Log parameters
        _log_params_to_mlflow(config)
    
        # Tag the simulator so runs logged by the old universe-average/random
        # fallback simulator can be told apart from signal-weighted runs
        mlflow.set_tags({
            "simulator": SIMULATOR_VERSION,
            "sim.rebalance_frequency": simulation_config["rebalance_frequency"],
            "sim.transaction_cost_bps": simulation_config["transaction_cost_bps"],
        })
    
    try:
        # Step 1: Load data
        logger.info("Loading price data...")
        price_data = _load_price_data(bundle_name, start_date, end_date)
        
        # Step 2: Apply preprocessing
        logger.info("Applying preprocessing steps...")
        processed_data = _apply_preprocessing(price_data, config.get("preprocessing", []))
        
        # Step 3: Calculate factors
        logger.info("Calculating factors...")
        factor_data = _apply_factors(processed_data, config.get("factors", []))
        
        # Step 4: Run algorithm
        logger.info("Running backtest algorithm...")
        algorithm_config = config.get("algorithm", {})
        performance = _run_algorithm(
            factor_data,
            algorithm_config,
            start_date,
            end_date,
            capital_base,
            simulation_config,
        )

        # Step 5: Calculate metrics
        logger.info("Calculating performance metrics...")
        metrics = calculate_all_metrics(performance)
        
        # Step 6: Save results
        results = {
            "performance": performance,
            "metrics": metrics,
            "config": config,
            "run_date": datetime.now().isoformat(),
        }
        
        # Save pickle file
        output_path = output_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {output_path}")
        
        # Log to MLflow
        if log_to_mlflow and MLFLOW_AVAILABLE:
            # Log metrics
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(name, value)
            
            # Log artifacts
            mlflow.log_artifact(str(output_path))
        
        logger.info("Backtest completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise
    
    finally:
        if log_to_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()


def _load_price_data(
    bundle_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load price data from bundle or database."""
    from qsconnect import Client
    
    client = Client()
    db_manager = client._db_manager
    
    prices = db_manager.get_prices(
        start_date=start_date,
        end_date=end_date,
    )
    
    if prices.is_empty():
        raise BacktestDataError(
            f"No price data between {start_date} and {end_date} in "
            f"{get_settings().duckdb_path}; ingest prices before running a backtest"
        )
    
    return prices.to_pandas()


def _apply_preprocessing(
    df: pd.DataFrame,
    preprocessing_config: list,
) -> pd.DataFrame:
    """Apply preprocessing steps from config."""
    from qsresearch.preprocessors import preprocess_price_data, universe_screener
    
    PREPROCESSING_FUNCS = {
        "price_preprocessor": preprocess_price_data,
        "universe_screener": universe_screener,
    }
    
    for step in preprocessing_config:
        func_name = step.get("name")  # Fixed: was "func", now "name" to match config
        params = step.get("params", {})
        
        if func_name in PREPROCESSING_FUNCS:
            func = PREPROCESSING_FUNCS[func_name]
            df = func(df, **params)
            logger.info(f"Applied preprocessing: {func_name}")
    
    return df


def _apply_factors(
    df: pd.DataFrame,
    factors_config: list,
) -> pd.DataFrame:
    """Apply factor calculations from config."""
    from qsresearch.features import FactorEngine
    
    engine = FactorEngine()
    
    for factor_spec in factors_config:
        name = factor_spec.get("name")
        params = factor_spec.get("params", {})
        
        if name:
            df = engine.calculate_factor(df, name, **params)
    
    return df


def _run_algorithm(
    df: pd.DataFrame,
    algorithm_config: Dict[str, Any],
    start_date: str,
    end_date: str,
    capital_base: float,
    simulation_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Run the trading algorithm.
    
    This is a simplified version. For production, integrate with Zipline Reloaded.
    """
    from qsresearch.strategies.factor import algorithms as algo_module
    
    algorithm_name = algorithm_config.get("callable", "use_factor_as_signal")
    # Configs give "module.path:function"; the function is looked up in algo_module
    algorithm_name = algorithm_name.rsplit(":", 1)[-1]
    params = algorithm_config.get("params", {})
    
    # Dynamically get the algorithm function
    if hasattr(algo_module, algorithm_name):
        algorithm_func = getattr(algo_module, algorithm_name)
    else:
        logger.warning(f"Algorithm '{algorithm_name}' not found, using default")
        algorithm_func = algo_module.use_factor_as_signal
    
    # Generate signals
    signals = algorithm_func(df, **params)
    
    # Simulate portfolio performance
    simulation_config = simulation_config or {}
    performance = _simulate_portfolio(
        signals,
        df,
        capital_base,
        start_date,
        end_date,
        rebalance_frequency=simulation_config.get("rebalance_frequency", DEFAULT_REBALANCE_FREQUENCY),
        transaction_cost_bps=simulation_config.get("transaction_cost_bps", DEFAULT_TRANSACTION_COST_BPS),
    )
    
    return performance


def _simulate_portfolio(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    capital_base: float,
    start_date: str,
    end_date: str,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """
    Simulate a portfolio that holds the signal weights.
    
    On each rebalance date (the first signal date of each period) the book is
    traded at that day's close to the signal weights for that date, so those
    weights earn the NEXT day's returns - no look-ahead. Between rebalances the
    positions drift with prices. Weights that sum to less than 1 leave the
    remainder in cash at zero return.
    
    Transaction costs are charged on each rebalance as
    turnover * transaction_cost_bps / 10,000, where turnover is the sum of
    |target weight - drifted weight| across symbols.
    
    Args:
        signals: Long-format frame with 'date', 'symbol' and 'weight' columns
        prices: Long-format frame with 'date', 'symbol' and 'close' columns
        capital_base: Starting portfolio value
        start_date: First date of the simulation window
        end_date: Last date of the simulation window
        rebalance_frequency: 'D' (every signal date), 'W', 'M' or 'Q'
        transaction_cost_bps: Cost per unit of turnover, in basis points
    
    Returns:
        DataFrame with date, portfolio_value, returns, turnover and
        transaction_cost (as a fraction of portfolio value) per trading day
    
    Raises:
        BacktestDataError: If prices or signals are missing or empty
    """
    import numpy as np

    if rebalance_frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(
            f"rebalance_frequency must be one of {sorted(REBALANCE_FREQUENCIES)}, "
            f"got {rebalance_frequency!r}"
        )
    if transaction_cost_bps < 0:
        raise ValueError(f"transaction_cost_bps must be >= 0, got {transaction_cost_bps}")

    if prices is None or prices.empty:
        raise BacktestDataError("Price data is empty; cannot simulate a portfolio")
    missing = {"date", "symbol", "close"} - set(prices.columns)
    if missing:
        raise BacktestDataError(f"Price data is missing required columns: {sorted(missing)}")
    if signals is None or signals.empty:
        raise BacktestDataError("Signals are empty; the algorithm selected no positions")
    missing = {"date", "symbol", "weight"} - set(signals.columns)
    if missing:
        raise BacktestDataError(f"Signals are missing required columns: {sorted(missing)}")
    
    # Wide close matrix (dates x symbols). Closes are forward-filled so a missing
    # bar is a zero return and the next bar carries the full move.
    px = prices[["date", "symbol", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.drop_duplicates(["date", "symbol"], keep="last")
    close = px.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    asset_returns = close.pct_change(fill_method=None).fillna(0.0)
    
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    asset_returns = asset_returns.loc[start:end]
    if asset_returns.empty:
        raise BacktestDataError(f"No price data between {start_date} and {end_date}")
    
    # Wide target-weight matrix on signal dates; symbols absent from a date get 0
    sig = signals[["date", "symbol", "weight"]].copy()
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.drop_duplicates(["date", "symbol"], keep="last")
    unknown = set(sig["symbol"]) - set(close.columns)
    if unknown:
        logger.warning(f"Dropping signals for {len(unknown)} symbols with no prices")
        sig = sig[sig["symbol"].isin(close.columns)]
    targets = (
        sig.pivot(index="date", columns="symbol", values="weight")
        .reindex(columns=close.columns)
        .fillna(0.0)
    )
    targets = targets[targets.index.isin(asset_returns.index)]
    if targets.empty:
        raise BacktestDataError(
            f"No signals fall on trading dates between {start_date} and {end_date}"
        )
    
    # Rebalance on the first signal date of each period
    if rebalance_frequency == "D":
        rebalance_dates = targets.index
    else:
        periods = targets.index.to_period(rebalance_frequency)
        rebalance_dates = targets.index[~periods.duplicated()]
    rebalance_set = set(rebalance_dates)
    
    cost_rate = transaction_cost_bps / 10_000
    returns_matrix = asset_returns.to_numpy()
    held = np.zeros(returns_matrix.shape[1])  # weights carried into the day
    
    port_returns = np.zeros(len(asset_returns))
    turnovers = np.zeros(len(asset_returns))
    costs = np.zeros(len(asset_returns))
    
    for i, date in enumerate(asset_returns.index):
        # 1. Weights set at the previous close earn today's returns
        day_ret = returns_matrix[i]
        gross = float(held @ day_ret)
        if gross > -1.0:
            held = held * (1.0 + day_ret) / (1.0 + gross)
        else:
            held = np.zeros_like(held)
    
        # 2. Trade to target at today's close; these weights earn tomorrow
        cost = 0.0
        if date in rebalance_set:
            target = targets.loc[date].to_numpy()
            turnovers[i] = float(np.abs(target - held).sum())
            cost = turnovers[i] * cost_rate
            held = target
        costs[i] = cost
        port_returns[i] = (1.0 + gross) * (1.0 - cost) - 1.0
    
    portfolio_value = capital_base * np.cumprod(1.0 + port_returns)
    
    logger.info(
        f"Simulated {len(asset_returns)} days, {len(rebalance_dates)} rebalances "
        f"({rebalance_frequency}), avg turnover {turnovers.sum() / len(rebalance_dates):.2%}, "
        f"cost {transaction_cost_bps} bps"
    )
    
    return pd.DataFrame({
        "date": asset_returns.index,
        "portfolio_value": portfolio_value,
        "returns": port_returns,
        "turnover": turnovers,
        "transaction_cost": costs,
    })


def _log_params_to_mlflow(config: Dict[str, Any]) -> None:
    """Log configuration parameters to MLflow."""
    def flatten_dict(d: Dict, parent_key: str = "") -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, str(v)[:250]))  # MLflow param limit
        return dict(items)
    
    flat_config = flatten_dict(config)
    
    for key, value in flat_config.items():
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Failed to log param {key}: {e}")
