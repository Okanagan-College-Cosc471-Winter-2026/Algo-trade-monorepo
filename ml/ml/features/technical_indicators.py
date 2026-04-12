"""
Technical Indicators for Stock Price Prediction

This module provides reusable functions for calculating technical indicators
used in feature engineering for stock price prediction models.
"""

import pandas as pd
import numpy as np


def calculate_sma(df: pd.DataFrame, windows: list[int] = [10, 20, 50]) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages and price distance from SMA.
    
    Args:
        df: DataFrame with 'close' column
        windows: List of window sizes for SMA calculation
        
    Returns:
        DataFrame with SMA columns and distance ratios
    """
    result = df.copy()
    
    for window in windows:
        sma_col = f'sma_{window}'
        dist_col = f'dist_sma_{window}'
        
        result[sma_col] = result['close'].rolling(window=window).mean()
        # Distance from SMA as percentage
        result[dist_col] = (result['close'] / result[sma_col]) - 1
    
    return result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period (default 14)
        
    Returns:
        DataFrame with 'rsi' column
    """
    result = df.copy()
    
    delta = result['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-8)  # Avoid division by zero
    result['rsi'] = 100 - (100 / (1 + rs))
    
    return result


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with 'macd', 'macd_signal', 'macd_diff' columns
    """
    result = df.copy()
    
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    
    result['macd'] = ema_fast - ema_slow
    result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
    result['macd_diff'] = result['macd'] - result['macd_signal']
    
    return result


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with 'close' column
        window: Rolling window size
        num_std: Number of standard deviations
        
    Returns:
        DataFrame with 'bb_upper', 'bb_lower', 'bb_width', 'bb_position' columns
    """
    result = df.copy()
    
    sma = result['close'].rolling(window=window).mean()
    std = result['close'].rolling(window=window).std()
    
    result['bb_upper'] = sma + (std * num_std)
    result['bb_lower'] = sma - (std * num_std)
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / sma
    # Position within bands (0 = lower, 0.5 = middle, 1 = upper)
    result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-8)
    
    return result


def calculate_returns(df: pd.DataFrame, lags: list[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Calculate lagged returns (percentage change).
    
    Args:
        df: DataFrame with 'close' column
        lags: List of lag periods
        
    Returns:
        DataFrame with 'ret_X' columns for each lag
    """
    result = df.copy()
    
    for lag in lags:
        result[f'ret_{lag}'] = result['close'].pct_change(lag)
    
    return result


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        df: DataFrame with 'close' column
        window: Rolling window size
        
    Returns:
        DataFrame with 'volatility' column
    """
    result = df.copy()
    
    returns = result['close'].pct_change()
    result[f'volatility_{window}'] = returns.rolling(window=window).std()
    
    return result


def calculate_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate volume-based features.
    
    Args:
        df: DataFrame with 'volume' column
        window: Rolling window size
        
    Returns:
        DataFrame with volume features
    """
    result = df.copy()
    
    result[f'vol_ma_{window}'] = result['volume'].rolling(window=window).mean()
    result['vol_ratio'] = result['volume'] / (result[f'vol_ma_{window}'] + 1)
    
    return result


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all technical indicators to create a complete feature set.
    
    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume, date)
        
    Returns:
        DataFrame with all technical indicators
    """
    result = df.copy()
    
    # Ensure date is datetime
    if 'date' in result.columns:
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').reset_index(drop=True)
    
    # Apply all indicators
    result = calculate_sma(result, windows=[10, 20, 50])
    result = calculate_rsi(result, period=14)
    result = calculate_macd(result)
    result = calculate_bollinger_bands(result, window=20)
    result = calculate_returns(result, lags=[1, 5, 10, 20])
    result = calculate_volatility(result, window=20)
    result = calculate_volume_features(result, window=20)
    
    # Add time features if date column exists
    if 'date' in result.columns:
        result['dayofweek'] = result['date'].dt.dayofweek
        result['month'] = result['date'].dt.month
    
    return result


def get_feature_columns() -> list[str]:
    """
    Get the list of feature column names used for model training.
    
    Returns:
        List of feature column names
    """
    return [
        # Lagged returns
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        # SMA distances
        'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
        # Momentum indicators
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        # Bollinger Bands
        'bb_width', 'bb_position',
        # Volatility
        'volatility_20',
        # Volume
        'vol_ratio',
        # Time features
        'dayofweek', 'month'
    ]
