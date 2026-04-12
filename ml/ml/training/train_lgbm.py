
import time
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sqlalchemy import create_engine, text
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
DB_CONFIG = {
    'host': '10.12.43.135',
    'port': 5432,
    'database': 'market_data',
    'user': 'mluser',
    'password': 'mlpassword'
}

STOCK_TICKER = 'aapl'
PREDICTION_HORIZON = 60  # Predict N periods ahead
TRAIN_RATIO = 0.8        # 80% train, 20% test
OUTPUT_DIR = './model_artifacts_lgbm'

# LightGBM Base Parameters
LGB_PARAMS = {
    'n_estimators': 300,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'objective': 'regression',
    'metric': 'rmse',
    'lambda_l1': 10,
    'lambda_l2': 1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

def get_db_engine():
    """Create SQLAlchemy engine for PostgreSQL connection."""
    connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(connection_string)

def load_stock_data(engine, ticker):
    """Load stock data from PostgreSQL database."""
    query = f'SELECT * FROM market."{ticker}" ORDER BY date'
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def create_technical_features(df):
    """
    Create technical indicator features from OHLCV data.
    """
    df = df.copy()

    # --- Basic Price Features ---
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['return_20'] = df['close'].pct_change(20)

    # --- Moving Averages ---
    for window in [5, 10, 20, 50, 100]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

    # --- Price relative to MAs ---
    df['close_to_sma_20'] = df['close'] / df['sma_20']
    df['close_to_sma_50'] = df['close'] / df['sma_50']
    df['sma_20_to_sma_50'] = df['sma_20'] / df['sma_50']

    # --- Volatility Features ---
    df['volatility_5'] = df['return_1'].rolling(window=5).std()
    df['volatility_10'] = df['return_1'].rolling(window=10).std()
    df['volatility_20'] = df['return_1'].rolling(window=20).std()

    # --- High-Low Range ---
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['hl_range_ma_10'] = df['hl_range'].rolling(window=10).mean()

    # --- Price Position within Range ---
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # --- Open-Close Relationship ---
    df['oc_range'] = (df['close'] - df['open']) / df['open']
    df['body_to_range'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)

    # --- Volume Features ---
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1)
    df['volume_change'] = df['volume'].pct_change(1)

    # --- RSI ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- Rate of Change ---
    df['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['roc_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

    # --- MACD ---
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- Bollinger Bands ---
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # --- Lag Features ---
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['return_1'].shift(lag)

    # --- Time-based Features ---
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)

    return df

def main():
    print(f"Starting LightGBM Training Pipeline for {STOCK_TICKER}...")
    
    # 1. Database Connection
    engine = get_db_engine()
    print("Database connection established!")

    # 2. Load Data
    print(f"Loading data for {STOCK_TICKER}...")
    df = load_stock_data(engine, STOCK_TICKER)
    print(f"Loaded: {len(df)} rows")

    # 3. Preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
    df = df.sort_values('date').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Feature Engineering
    print("Engineering features...")
    df = create_technical_features(df)
    
    # Create target
    df['target'] = df['close'].shift(-PREDICTION_HORIZON)
    
    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    print(f"After cleaning: {len(df)} rows")

    # 5. Split Data
    exclude_cols = ['date', 'target', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # 6. Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Hyperparameter Tuning
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        subsample_freq=1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    param_dist = {
        'n_estimators': [100, 300, 500, 700],
        'num_leaves': [15, 31, 63, 127, 255],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'lambda_l1': [0, 1, 10, 50],
        'lambda_l2': [0, 1, 5, 10],
        'min_child_samples': [5, 10, 20, 50],
        'min_split_gain': [0, 0.01, 0.1, 0.5]
    }

    print("Starting hyperparameter tuning (RandomizedSearchCV)...")
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    print(f"Tuning completed in {time.time() - start_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV RMSE: {-random_search.best_score_:.4f}")

    model = random_search.best_estimator_

    # 8. Evaluation
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Train MAPE: {train_mape:.2f}%")
    print(f"Test MAPE:  {test_mape:.2f}%")

    # 9. Save Artifacts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save Model
    model.booster_.save_model(os.path.join(OUTPUT_DIR, "lightgbm_model.txt"))
    joblib.dump(model, os.path.join(OUTPUT_DIR, "lightgbm_model.joblib"))
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    
    # Save Features
    joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_names.joblib"))
    
    # Save Config
    config = {
        'prediction_horizon': PREDICTION_HORIZON,
        'train_ratio': TRAIN_RATIO,
        'model_type': 'lightgbm'
    }
    joblib.dump(config, os.path.join(OUTPUT_DIR, "config.joblib"))
    
    # Save Predictions
    results = pd.DataFrame({
        'datetime': test_df['date'].values,
        'actual': y_test.values,
        'predicted': y_pred_test,
        'difference': np.abs(y_test.values - y_pred_test)
    })
    results.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    
    print(f"\nArtifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
