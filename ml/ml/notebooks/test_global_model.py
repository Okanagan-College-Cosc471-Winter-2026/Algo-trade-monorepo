#!/usr/bin/env python
# coding: utf-8

# # Test Global Model Pipeline
# Quick validation with a small subset before full training

# In[1]:


import os
import time
import joblib
import warnings
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ## Configuration

# In[ ]:




# In[2]:


# Database
DB_CONFIG = {
    'host': '10.12.43.135',
    'port': 5432,
    'database': 'market_data',
    'user': 'mluser',
    'password': 'mlpassword'
}

PREDICTION_HORIZON = 60  # 5 hours ahead (60 * 5min)

# ## Database Functions

# In[3]:


import pandas as pd
from sqlalchemy import create_engine

def get_db_engine():
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def load_stock_data(engine, ticker):
    # UPDATED to use the correct schema and table from analysis
    query = f"SELECT ts as date, open, high, low, close, volume FROM stg_transform.market_data WHERE symbol = '{ticker}' ORDER BY ts"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

engine = get_db_engine()
print("Connected to database")

# ## Feature Engineering

# In[4]:


def create_features(df, ticker_id):
    """
    Generate extensive technical indicators and targets.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Target: Future Return
    df['target'] = df['close'].shift(-PREDICTION_HORIZON) / df['close'] - 1
    
    # Returns (momentum over different time frames)
    for lag in [1, 5, 10, 20]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag)
    
    # Moving Averages (SMA & EMA)
    for window in [10, 20, 50, 100]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'dist_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
        # Exponential MA
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        df[f'dist_ema_{window}'] = df['close'] / df[f'ema_{window}'] - 1
        
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volatility / Bollinger Bands
    for window in [10, 20]:
        df[f'volatility_{window}'] = df['ret_1'].rolling(window=window).std()
        # BB Width
        rolling_std = df['close'].rolling(window=window).std()
        df[f'bb_{window}_width'] = (rolling_std * 4) / df[f'sma_{window}']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume Indicators
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma_20'] + 1)
    
    # Price Extremes (Min/Max Last X Periods)
    df['rolling_max_10'] = df['high'].rolling(10).max()
    df['dist_roll_max_10'] = df['close'] / df['rolling_max_10'] - 1
    df['rolling_min_10'] = df['low'].rolling(10).min()
    df['dist_roll_min_10'] = df['close'] / df['rolling_min_10'] - 1
    
    # Time Features
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day
    
    # Ticker ID
    df['ticker_id'] = ticker_id
    
    # Cleanup
    df = df.dropna().reset_index(drop=True)
    
    # Downcast to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    
    feature_cols = [
        'ticker_id', 'hour', 'dayofweek', 'month', 'dayofmonth',
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'dist_sma_10', 'dist_sma_50', 'dist_sma_20', 'dist_sma_100',
        'dist_ema_10', 'dist_ema_50', 'dist_ema_20', 'dist_ema_100',
        'macd', 'macd_signal', 'macd_hist',
        'volatility_10', 'volatility_20', 
        'bb_10_width', 'bb_20_width',
        'rsi', 'vol_ratio',
        'dist_roll_max_10', 'dist_roll_min_10'
    ]
    
    return df[feature_cols], df['target'], df['date']

# ## Load All Stocks

# In[5]:


# Fetch all available symbols from DB
with engine.connect() as conn:
    all_symbols = pd.read_sql("SELECT DISTINCT symbol FROM stg_transform.market_data ORDER BY symbol", conn)['symbol'].tolist()

print(f"Found {len(all_symbols)} symbols: {all_symbols}")

# Label Encoder
le = LabelEncoder()
le.fit(all_symbols)

all_X = []
all_y = []
all_dates = []

for ticker in all_symbols:
    print(f"Loading {ticker}...")
    df_raw = load_stock_data(engine, ticker)
    print(f"  Raw rows: {len(df_raw)}")
    
    tid = le.transform([ticker])[0]
    X, y, dates = create_features(df_raw, tid)
    print(f"  After features: {len(X)} rows")
    
    all_X.append(X)
    all_y.append(y)
    all_dates.append(dates)

X_full = pd.concat(all_X, ignore_index=True)
y_full = pd.concat(all_y, ignore_index=True)
dates_full = pd.concat(all_dates, ignore_index=True)

print(f"\nTotal: {len(X_full)} rows")

# ## Inspect Data

# In[6]:


print("Features:")
print(X_full.columns.tolist())
print(f"\nShape: {X_full.shape}")
print(f"\nFeature Stats:")
X_full.describe()

# In[7]:


print("Target (future returns) Stats:")
print(f"  Mean:  {y_full.mean()*100:.4f}%")
print(f"  Std:   {y_full.std()*100:.4f}%")
print(f"  Min:   {y_full.min()*100:.4f}%")
print(f"  Max:   {y_full.max()*100:.4f}%")

plt.figure(figsize=(10, 4))
plt.hist(y_full, bins=100, edgecolor='black', alpha=0.7)
plt.title('Target Distribution (Future Returns)')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--')
plt.show()

# ## Train/Test Split

# In[8]:


split_date = dates_full.quantile(0.8)
print(f"Split date: {split_date}")

mask_train = dates_full <= split_date
mask_test = dates_full > split_date

X_train = X_full[mask_train]
y_train = y_full[mask_train]
X_test = X_full[mask_test]
y_test = y_full[mask_test]

print(f"Train: {len(X_train)} rows")
print(f"Test:  {len(X_test)} rows")

# ## Train Model (Quick Test)

# In[9]:


# GPU-accelerated model for faster training on H100 with massive estimators
model = XGBRegressor(
    n_estimators=3000,  # Maximize compute with a large number of trees
    learning_rate=0.01, # Smaller learning rate since we have more trees
    max_depth=9,        # Deeper trees to capture complex market patterns
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    device='cuda',      # properly utilize H100 resources
    objective='reg:squarederror',
    random_state=42,
    early_stopping_rounds=100
)

print("Training...")
start = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
print(f"\nTraining time: {time.time()-start:.1f}s")

# ## Evaluate

# In[10]:


y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("=" * 40)
print("TEST RESULTS")
print("=" * 40)
print(f"RMSE: {rmse:.6f} ({rmse*100:.4f}%)")
print(f"MAE:  {mae:.6f} ({mae*100:.4f}%)")

# In[11]:


# Actual vs Predicted scatter
plt.figure(figsize=(10, 5))

# Sample for plotting (too many points)
sample_idx = np.random.choice(len(y_test), size=min(5000, len(y_test)), replace=False)

plt.scatter(y_test.iloc[sample_idx], y_pred[sample_idx], alpha=0.3, s=1)
plt.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect prediction')
plt.xlabel('Actual Return')
plt.ylabel('Predicted Return')
plt.title('Actual vs Predicted Returns')
plt.legend()
plt.xlim(-0.1, 0.1)
plt.ylim(-0.1, 0.1)
plt.grid(True, alpha=0.3)
plt.show()

# In[12]:


# Feature Importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# ## Direction Accuracy
# How often does the model predict the correct direction (up/down)?

# In[13]:


# Direction accuracy
actual_direction = (y_test > 0).astype(int)
pred_direction = (y_pred > 0).astype(int)

direction_accuracy = (actual_direction == pred_direction).mean()
print(f"Direction Accuracy: {direction_accuracy*100:.2f}%")
print(f"(Random baseline: 50%)")

# ## Conclusion
# 
# If results look reasonable, run the full training script:
# ```bash
# conda activate cosc471
# python scripts/train_global_model.py
# ```

# In[14]:


import joblib
import os

# Create models directory if not exists
os.makedirs('../models', exist_ok=True)

model_path = '../models/global_xgb_gpu.joblib'
print(f"Saving trained model to {model_path}...")
joblib.dump(model, model_path)

file_size = os.path.getsize(model_path) / (1024*1024)
print(f"Saved! File size: {file_size:.2f} MB")
