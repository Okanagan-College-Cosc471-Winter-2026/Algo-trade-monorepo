#!/usr/bin/env python
"""
Global Stock Prediction Model Training Script

Requirements (install via conda):
    conda activate cosc471
    # or: pip install pandas numpy sqlalchemy psycopg2-binary scikit-learn xgboost joblib

Usage:
    conda activate cosc471
    python scripts/train_global_model.py
"""

import os
import time
import joblib
import warnings
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
DB_CONFIG = {
    'host': '10.12.43.135',
    'port': 5432,
    'database': 'app',
    'user': 'postgres',
    'password': 'changethis'
}

OUTPUT_DIR = './model_artifacts_global'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Parameters
PREDICTION_HORIZON = 60  # Predict return 60 periods ahead
N_JOBS = 28             # Use 28 cores as requested

# ===== DATABASE FUNCTIONS =====

def get_db_engine():
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def get_all_stocks(engine):
    """Get list of all stock symbols from the unified stocks table."""
    query = "SELECT symbol FROM market.stocks WHERE is_active = true ORDER BY symbol"
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    return result['symbol'].tolist()

def load_stock_data(engine, symbol):
    """Load stock data from the unified daily_prices table."""
    query = f"""
        SELECT date, open, high, low, close, volume, previous_close, change, change_pct
        FROM market.daily_prices 
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

# ===== FEATURE ENGINEERING =====

def create_features(df, ticker_id):
    """
    Generate technical indicators and targets efficiently.
    converts types to float32 to save memory.
    """
    # 1. Setup
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. Target: Future Return (Predicted Variable)
    # We predict the percentage change PREDICTION_HORIZON steps ahead
    df['target'] = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    
    # 3. Technical Indicators
    
    # Returns (Lags)
    for lag in [1, 5, 10, 20]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag)
    
    # Moving Averages
    for window in [10, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        # Price relative to SMA (Normalized)
        df[f'dist_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
        
    # Volatility (Standard Deviation of returns)
    df['volatility_20'] = df['ret_1'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume Features
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma_20'] + 1)
    
    # Time Features
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Identity Feature
    df['ticker_id'] = ticker_id
    
    # 4. Cleanup & Memory Optimization
    # Drop rows with NaN (due to rolling/shifting)
    df = df.dropna().reset_index(drop=True)
    
    # Downcast to float32 to save RAM
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    
    # Select features for training
    feature_cols = [
        'ticker_id', 'hour', 'dayofweek',
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'dist_sma_10', 'dist_sma_50',
        'volatility_20', 'rsi', 'vol_ratio'
    ]
    
    return df[feature_cols], df['target'], df['date']

# ===== MAIN PIPELINE =====

def main():
    print(f"Starting Global Model Training (Jobs: {N_JOBS})...")
    start_time = time.time()
    
    engine = get_db_engine()
    symbols = get_all_stocks(engine)
    
    print(f"Found {len(symbols)} active stocks in database: {', '.join(symbols)}")
    
    # 1. Data Ingestion & Processing
    all_X = []
    all_y = []
    all_dates = []
    
    # Label Encoder for Symbols (mapped to integers)
    le = LabelEncoder()
    le.fit(symbols)
    joblib.dump(le, f"{OUTPUT_DIR}/ticker_encoder.joblib")
    
    # We load stocks in batches to monitor progress
    print("Processing stocks...")
    for i, symbol in enumerate(symbols):
        try:
            # Load raw
            df_raw = load_stock_data(engine, symbol)
            
            # Encode symbol string to int
            sid = le.transform([symbol])[0]
            
            # Featurize
            X, y, dates = create_features(df_raw, sid)
            
            # Append to lists (efficient memory collection)
            all_X.append(X)
            all_y.append(y)
            all_dates.append(dates)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(symbols)} stocks...")
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    print("Concatenating datasets...")
    # Concatenate all dataframes once
    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)
    dates_full = pd.concat(all_dates, ignore_index=True)
    
    # Free up list memory
    del all_X, all_y, all_dates
    
    print(f"Total Dataset Size: {len(X_full)} rows")
    print(f"Features: {X_full.columns.tolist()}")
    
    # 2. Train/Test Split (Time Based)
    # Since we have multiple stocks, a simple time split works if dates are aligned.
    # We split by DATE, not by row index, to prevent leakage.
    
    split_date = dates_full.quantile(0.8) # 80% train, 20% test
    print(f"Splitting data at {split_date}")
    
    mask_train = dates_full <= split_date
    mask_test = dates_full > split_date
    
    X_train = X_full[mask_train]
    y_train = y_full[mask_train]
    X_test = X_full[mask_test]
    y_test = y_full[mask_test]
    
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape:  {X_test.shape}")
    
    # 3. XGBoost Model Training
    print("Initializing XGBoost...")
    model = XGBRegressor(
        n_estimators=1000,        # Increased from 500
        learning_rate=0.01,       # Reduced from 0.05 (slower learning = more trees)
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=N_JOBS,            # 28 Cores
        tree_method='hist',       # Faster histogram optimization
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=50  # Increased patience from 20
    )
    
    print("Training model (this may take a while)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # 4. Evaluation
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # MAE is often better for returns since they are small decimals
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE:  {mae:.6f}")
    
    # 5. Save Artifacts
    model.save_model(f"{OUTPUT_DIR}/global_model.json")
    print(f"Model saved to {OUTPUT_DIR}/global_model.json")
    
    print(f"Total execution time: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()
