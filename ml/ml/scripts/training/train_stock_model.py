#!/usr/bin/env python3
"""
Stock Prediction Model Training Script

Trains an XGBoost model to predict stock price movements using
historical OHLC data from PostgreSQL.

Usage:
    python ml/scripts/train_stock_model.py
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib
from dotenv import load_dotenv

# Add parent directory to path to import features module
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.technical_indicators import create_all_features, get_feature_columns

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====

# Database configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_SERVER', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'market_data'),
    'user': os.getenv('POSTGRES_USER', 'mluser'),
    'password': os.getenv('POSTGRES_PASSWORD', 'mlpassword')
}

# Model configuration
MODEL_CONFIG = {
    'prediction_horizon': 26,  # Predict 1 day ahead (26 bars @ 15min intervals)
    'train_test_split': 0.8,  # 80% train, 20% test
    'random_state': 42
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 3000,
    'max_depth': 9,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': MODEL_CONFIG['random_state'],
    'n_jobs': -1,  # Use all available cores
    'tree_method': 'hist',  # Faster training
    'device': 'cuda'
}

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'models' / 'stock_prediction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# ===== DATABASE FUNCTIONS =====

def get_db_engine():
    """Create SQLAlchemy engine for PostgreSQL connection."""
    connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(connection_string)


def load_stock_data(engine) -> pd.DataFrame:
    """
    Load all stock data from PostgreSQL.
    
    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, volume
    """
    query = """
    SELECT 
        dp.symbol,
        dp.ts as date,
        dp.open,
        dp.high,
        dp.low,
        dp.close,
        dp.volume
    FROM stg_transform.market_data dp
    ORDER BY dp.symbol, dp.ts
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    print(f"Loaded {len(df):,} records for {df['symbol'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


# ===== FEATURE ENGINEERING =====

def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare dataset with features and target variable.
    
    Args:
        df: Raw OHLCV data
        
    Returns:
        Tuple of (features_df, target_series)
    """
    print("\nEngineering features...")
    
    all_features = []
    all_targets = []
    all_dates = []
    all_symbols = []
    
    # Process each stock separately
    for symbol in df['symbol'].unique():
        stock_df = df[df['symbol'] == symbol].copy()
        
        # Skip stocks with insufficient data
        if len(stock_df) < 100:
            continue
        
        # Create features
        stock_features = create_all_features(stock_df)
        
        # Create target: next-day return
        horizon = MODEL_CONFIG['prediction_horizon']
        stock_features['target'] = stock_features['close'].shift(-horizon) / stock_features['close'] - 1
        
        # Add symbol identifier
        stock_features['symbol'] = symbol
        
        # Drop rows with NaN (from rolling windows and target shift)
        stock_features = stock_features.dropna()
        
        if len(stock_features) > 0:
            all_features.append(stock_features)
    
    # Concatenate all stocks
    full_df = pd.concat(all_features, ignore_index=True)
    
    print(f"Total samples after feature engineering: {len(full_df):,}")
    print(f"Features per sample: {len(get_feature_columns())}")
    
    # Separate features and target
    feature_cols = get_feature_columns()
    X = full_df[feature_cols]
    y = full_df['target']
    dates = full_df['date']
    symbols = full_df['symbol']
    
    return X, y, dates, symbols


# ===== TRAINING =====

def train_test_split_temporal(X, y, dates, split_ratio=0.8):
    """
    Split data chronologically to prevent data leakage.
    
    Args:
        X: Features
        y: Target
        dates: Date column
        split_ratio: Train/test split ratio
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_date = dates.quantile(split_ratio)
    
    train_mask = dates <= split_date
    test_mask = dates > split_date
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\nTrain/Test Split:")
    print(f"  Split date: {split_date}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model with early stopping.
    
    Returns:
        Trained model
    """
    print("\nTraining XGBoost model...")
    print(f"Hyperparameters: {XGBOOST_PARAMS}")
    
    model = XGBRegressor(**XGBOOST_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    print(f"\nTraining complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Directional accuracy (did we predict the direction correctly?)
    direction_actual = (y_test > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    directional_accuracy = (direction_actual == direction_pred).mean()
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': float(directional_accuracy)
    }
    
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Directional Accuracy: {directional_accuracy:.2%}")
    
    return metrics


# ===== SAVE ARTIFACTS =====

def save_artifacts(model, metrics):
    """
    Save model and metadata to disk.
    """
    print(f"\nSaving artifacts to {OUTPUT_DIR}...")
    
    # Save XGBoost model
    model_path = OUTPUT_DIR / 'model.json'
    model.save_model(str(model_path))
    print(f"  ✓ Saved model to {model_path}")
    
    # Save feature names
    feature_names_path = OUTPUT_DIR / 'feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(get_feature_columns(), f, indent=2)
    print(f"  ✓ Saved feature names to {feature_names_path}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': get_feature_columns(),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = OUTPUT_DIR / 'feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"  ✓ Saved feature importance to {importance_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'XGBoost Regressor',
        'prediction_horizon': MODEL_CONFIG['prediction_horizon'],
        'hyperparameters': XGBOOST_PARAMS,
        'metrics': metrics,
        'feature_count': len(get_feature_columns())
    }
    
    metadata_path = OUTPUT_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    print("\n✅ All artifacts saved successfully!")


# ===== MAIN =====

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Stock Prediction Model Training")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/5] Loading data from PostgreSQL...")
    engine = get_db_engine()
    df = load_stock_data(engine)
    
    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    X, y, dates, symbols = prepare_dataset(df)
    
    # 3. Train/test split
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split_temporal(
        X, y, dates, 
        split_ratio=MODEL_CONFIG['train_test_split']
    )
    
    # 4. Train model
    print("\n[4/5] Training model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    # 5. Evaluate and save
    print("\n[5/5] Evaluating and saving...")
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
