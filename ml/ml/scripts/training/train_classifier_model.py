#!/usr/bin/env python
"""
Stock Direction Classification Model (Split-on-Ingest Optimization)

Predicts: UP / NEUTRAL / DOWN (1-hour ahead)
- Uses market-relative features (stock vs SPY)
- Classification instead of regression
- Shorter prediction horizon (1 hour = 12 periods)

Usage:
    conda activate cosc471
    python scripts/train_classifier_model.py
"""

import os
import time
import joblib
import warnings
import gc
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
DB_CONFIG = {
    'host': '10.12.43.135',
    'port': 5432,
    'database': 'market_data',
    'user': 'mluser',
    'password': 'mlpassword'
}

OUTPUT_DIR = './model_artifacts_classifier'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Parameters
PREDICTION_HORIZON = 12    # 1 hour ahead (12 * 5min)
THRESHOLD_UP = 0.003       # +0.3% = UP
THRESHOLD_DOWN = -0.003    # -0.3% = DOWN
N_JOBS = 16                # Conservative core usage

# Market reference ticker
MARKET_TICKER = 'spy'

# ===== DATABASE FUNCTIONS =====

def get_db_engine():
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def get_all_tables(engine):
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'market'
    ORDER BY table_name
    """
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    return result['table_name'].tolist()

def load_stock_data(engine, ticker, columns=None):
    if columns:
        cols_str = ", ".join([f'"{c}"' for c in columns])
        query = f'SELECT {cols_str} FROM market."{ticker}" ORDER BY date'
    else:
        query = f'SELECT date, open, high, low, close, volume FROM market."{ticker}" ORDER BY date'
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

# ===== FEATURE ENGINEERING =====

def create_features(df, ticker_id, market_df=None):
    """
    Generate features with market-relative indicators.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # === TARGET: Classification ===
    future_return = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    
    # Initialize target with NEUTRAL (1)
    df['target'] = 1
    df.loc[future_return > THRESHOLD_UP, 'target'] = 2      # UP
    df.loc[future_return < THRESHOLD_DOWN, 'target'] = 0    # DOWN
    df['target'] = df['target'].astype(np.int8)

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

    df['rsi_oversold'] = (df['rsi'] < 30).astype(np.int8)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(np.int8)

    # === VOLUME ===
    df['vol_ma_12'] = df['volume'].rolling(window=12).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma_12'] + 1)
    df['vol_spike'] = (df['vol_ratio'] > 2).astype(np.int8)

    # === TIME FEATURES ===
    df['hour'] = df['date'].dt.hour.astype(np.int8)
    df['minute'] = df['date'].dt.minute.astype(np.int8)
    df['dayofweek'] = df['date'].dt.dayofweek.astype(np.int8)

    df['is_open_hour'] = ((df['hour'] == 9) & (df['minute'] < 60)).astype(np.int8)
    df['is_close_hour'] = (df['hour'] >= 15).astype(np.int8)

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

    # === CLEANUP ===
    df = df.dropna().reset_index(drop=True)

    # Downcast floats
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

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

    return df[feature_cols], df['target'], df['date']

# ===== MAIN PIPELINE =====

def main():
    print("=" * 60)
    print("STOCK DIRECTION CLASSIFIER (SPLIT-ON-INGEST)")
    print("=" * 60)
    print(f"Using {N_JOBS} CPU cores context")
    
    start_time = time.time()
    engine = get_db_engine()
    tables = get_all_tables(engine)

    # Filter non-stock tables
    exclude_tables = {'abcde', 'modelfeatures', 'sectors', 'stock_metadata'}
    tables = [t for t in tables if t not in exclude_tables]
    print(f"Found {len(tables)} stocks")

    # 1. Determine Split Date globally (Lightweight Pass)
    print("Determining split date (scanning dates)...")
    all_dates = []
    # Sample every 10th table to estimate split date quickly and safely
    sample_tables = tables[::10] 
    for t in sample_tables:
        try:
            d = load_stock_data(engine, t, columns=['date'])
            all_dates.append(d['date'])
        except:
            pass
    
    global_dates = pd.concat(all_dates)
    global_dates = pd.to_datetime(global_dates)
    
    split_date = global_dates.quantile(0.8)
    print(f"Global Split Date (80%): {split_date}")
    
    del all_dates, global_dates
    gc.collect()

    # Load market data
    print(f"Loading market data ({MARKET_TICKER.upper()})...")
    try:
        market_df = load_stock_data(engine, MARKET_TICKER)
    except:
        market_df = None

    le = LabelEncoder()
    le.fit(tables)
    joblib.dump(le, f"{OUTPUT_DIR}/ticker_encoder.joblib")

    # 2. Main processing Loop (Split immediately)
    train_X_list, train_y_list = [], []
    test_X_list, test_y_list = [], []
    
    # Pre-allocate counters
    train_rows = 0
    test_rows = 0

    print("\nProcessing stocks and splitting on-the-fly...")
    for i, ticker in enumerate(tables):
        try:
            df_raw = load_stock_data(engine, ticker)
            tid = le.transform([ticker])[0]
            X, y, dates = create_features(df_raw, tid, market_df)
            
            # Split immediately
            mask_train = dates <= split_date
            mask_test = dates > split_date
            
            # Append subsets
            if mask_train.any():
                X_tr = X[mask_train]
                y_tr = y[mask_train]
                train_X_list.append(X_tr)
                train_y_list.append(y_tr)
                train_rows += len(X_tr)
                
            if mask_test.any():
                X_te = X[mask_test]
                y_te = y[mask_test]
                test_X_list.append(X_te)
                test_y_list.append(y_te)
                test_rows += len(X_te)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(tables)} stocks. Train: {train_rows:,}, Test: {test_rows:,}")
                gc.collect()

        except Exception as e:
            continue

    print("\nConstructing final datasets...")
    # Concatenate Train
    X_train = pd.concat(train_X_list, ignore_index=True)
    y_train = pd.concat(train_y_list, ignore_index=True)
    del train_X_list, train_y_list
    gc.collect()
    
    # Concatenate Test
    X_test = pd.concat(test_X_list, ignore_index=True)
    y_test = pd.concat(test_y_list, ignore_index=True)
    del test_X_list, test_y_list
    gc.collect()
    
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape:  {X_test.shape}")

    # Calculate Weights
    print("Calculating weights...")
    class_weights = {}
    total = len(y_train)
    for cls in [0, 1, 2]:
        count = (y_train == cls).sum()
        class_weights[cls] = total / (3 * count) if count > 0 else 1.0
    
    # Map weights
    sample_weights = y_train.map(class_weights).values.astype(np.float32)

    # Train
    print("\nInitializing XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=N_JOBS,
        tree_method='hist',
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        early_stopping_rounds=30
    )

    print("Training model...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluation
    print("Generating predictions...")
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'NEUTRAL', 'UP']))

    # Save
    model.save_model(f"{OUTPUT_DIR}/classifier_model.json")
    print(f"\nModel saved to {OUTPUT_DIR}/")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()
