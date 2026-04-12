#!/usr/bin/env python
"""
Inference Test Script for Stock Direction Classifier

Usage:
    conda activate cosc471
    python scripts/test_inference.py --ticker aapl
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBClassifier, DMatrix

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
DB_CONFIG = {
    'host': '10.12.43.135',
    'port': 5432,
    'database': 'market_data',
    'user': 'mluser',
    'password': 'mlpassword'
}
ARTIFACTS_DIR = './model_artifacts_classifier'
MARKET_TICKER = 'spy'

# ===== FEATURE ENGINEERING (Must match training logic) =====
def create_features_inference(df, ticker_id, market_df=None):
    """
    Generate features for the latest data points.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # We don't need 'target' for inference, just features
    
    # === PRICE FEATURES ===
    for lag in [1, 3, 6, 12]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag)

    df['momentum_12'] = df['close'] / df['close'].shift(12) - 1
    df['momentum_36'] = df['close'] / df['close'].shift(36) - 1

    for window in [12, 36, 72]:
        sma = df['close'].rolling(window=window).mean()
        df[f'dist_sma_{window}'] = df['close'] / sma - 1

    # === VOLATILITY ===
    df['volatility_12'] = df['ret_1'].rolling(window=12).std()
    df['volatility_36'] = df['ret_1'].rolling(window=36).std()

    # High-Low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['hl_range_ma'] = df['hl_range'].rolling(window=12).mean()

    # === RSI ===
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

    # === VOLUME ===
    df['vol_ma_12'] = df['volume'].rolling(window=12).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma_12'] + 1)
    df['vol_spike'] = (df['vol_ratio'] > 2).astype(int)

    # === TIME FEATURES ===
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['dayofweek'] = df['date'].dt.dayofweek

    df['is_open_hour'] = ((df['hour'] == 9) & (df['minute'] < 60)).astype(int)
    df['is_close_hour'] = (df['hour'] >= 15).astype(int)

    # === MARKET-RELATIVE FEATURES ===
    if market_df is not None:
        market_df = market_df.copy()
        market_df['date'] = pd.to_datetime(market_df['date'])
        market_df = market_df.sort_values('date').set_index('date')
        
        market_df['market_ret_1'] = market_df['close'].pct_change(1)
        market_df['market_ret_12'] = market_df['close'].pct_change(12)
        market_df['market_momentum'] = market_df['close'] / market_df['close'].shift(12) - 1

        df = df.set_index('date')
        df = df.join(market_df[['market_ret_1', 'market_ret_12', 'market_momentum']], how='left')
        df = df.reset_index()

        df['rel_strength'] = df['ret_12'] - df['market_ret_12']
        df['rel_strength'] = df['rel_strength'].fillna(0)
    else:
        df['market_ret_1'] = 0
        df['market_ret_12'] = 0
        df['market_momentum'] = 0
        df['rel_strength'] = 0

    # === TICKER ID ===
    df['ticker_id'] = ticker_id

    # Feature columns (Same as training)
    feature_cols = [
        'ticker_id', 'hour', 'dayofweek',
        'ret_1', 'ret_3', 'ret_6', 'ret_12',
        'momentum_12', 'momentum_36',
        'dist_sma_12', 'dist_sma_36', 'dist_sma_72',
        'volatility_12', 'volatility_36',
        'hl_range', 'hl_range_ma',
        'rsi', 'rsi_oversold', 'rsi_overbought',
        'vol_ratio', 'vol_spike',
        'is_open_hour', 'is_close_hour',
        'market_ret_1', 'market_ret_12', 'market_momentum', 'rel_strength'
    ]
    
    # Return features for the last row (latest state)
    # We take the last few rows to cover the rolling window requirements, then return the last one
    return df.iloc[[-1]][feature_cols], df.iloc[[-1]]['date'].values[0]

# ===== MAIN =====

def get_db_engine():
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def load_recent_data(engine, ticker, limit=200):
    query = f'SELECT date, open, high, low, close, volume FROM market."{ticker}" ORDER BY date DESC LIMIT {limit}'
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df.sort_values('date')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="aapl", help="Ticker symbol to predict")
    args = parser.parse_args()
    
    ticker = args.ticker.lower()
    
    print(f"Loading artifacts from {ARTIFACTS_DIR}...")
    try:
        le = joblib.load(f"{ARTIFACTS_DIR}/ticker_encoder.joblib")
        config = joblib.load(f"{ARTIFACTS_DIR}/config.joblib")
        model = XGBClassifier()
        model.load_model(f"{ARTIFACTS_DIR}/classifier_model.json")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        print("Has training completed successfully?")
        return

    engine = get_db_engine()
    
    print(f"Fetching data for {ticker.upper()}...")
    try:
        df = load_recent_data(engine, ticker)
        
        # Determine ticker ID
        try:
            tid = le.transform([ticker])[0]
        except:
            print(f"Ticker {ticker} was not seen during training. Using random ID 0.")
            tid = 0
            
        # Get Market data
        try:
            market_df = load_recent_data(engine, MARKET_TICKER)
        except:
            market_df = None

        print("Generating features...")
        X_latest, current_timestamp = create_features_inference(df, tid, market_df)
        
        print(f"\nPrediction time: {pd.to_datetime(current_timestamp)}")
        print(f"Current Price:   ${df.iloc[-1]['close']:.2f}")
        
        # Predict
        probs = model.predict_proba(X_latest)[0]
        pred_class = np.argmax(probs)
        
        class_names = config['class_names']
        direction = class_names[pred_class]
        
        print("\n" + "-"*30)
        print(f"PREDICTION: {direction}")
        print("-" * 30)
        print(f"Probability DOWN:    {probs[0]*100:.1f}%")
        print(f"Probability NEUTRAL: {probs[1]*100:.1f}%")
        print(f"Probability UP:      {probs[2]*100:.1f}%")
        print("-" * 30)
        
        print(f"\nHorizon: {config['prediction_horizon'] * 5} minutes")
        print(f"Thresholds: +/- {config['threshold_up']*100}%")

    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
