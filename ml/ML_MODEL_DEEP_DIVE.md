# ML Model Deep Dive — XGBoost Intraday Warm-Refresh System

Everything you need to understand the ML brain of this trading system: what it predicts, why XGBoost, how it's built, the math behind key decisions, and how each piece connects to production.

---

## Table of Contents

1. [What the model predicts](#1-what-the-model-predicts)
2. [Why XGBoost, not a neural network](#2-why-xgboost-not-a-neural-network)
3. [The 26-booster architecture](#3-the-26-booster-architecture)
4. [Feature engineering](#4-feature-engineering)
5. [Target definition — why log-returns](#5-target-definition--why-log-returns)
6. [Warm-refresh — updating without forgetting](#6-warm-refresh--updating-without-forgetting)
7. [Training pipeline — base model](#7-training-pipeline--base-model)
8. [Walk-forward validation](#8-walk-forward-validation)
9. [Hyperparameters — chosen values and why](#9-hyperparameters--chosen-values-and-why)
10. [Multi-GPU parallelism on H100](#10-multi-gpu-parallelism-on-h100)
11. [Inference — how a prediction is made](#11-inference--how-a-prediction-is-made)
12. [Model bundle on disk](#12-model-bundle-on-disk)
13. [Key ML concepts explained](#13-key-ml-concepts-explained)

---

## 1. What the model predicts

**Task**: Given all market data available up to a given 15-minute bar during the trading day, predict the closing price at each remaining 15-minute bar for the rest of that day.

**Concretely**: 8 symbols (AAPL, AMD, AMZN, BA, MSFT, NVDA, GOOGL, TSLA), 26 bars per day (09:30–15:45 ET), 15-minute resolution.

**Output**: For each symbol, 26 numbers. Each number is a **log-return** relative to the current close:

```
target_h00 = log( close_at_09:30 / close_now )
target_h01 = log( close_at_09:45 / close_now )
...
target_h25 = log( close_at_15:45 / close_now )
```

The frontend converts these back to absolute prices:
```python
pred_close_i = current_price * exp(log_return_i)
```

This gives you the full predicted intraday price path — not just "up or down" but a complete trajectory through the trading day.

---

## 2. Why XGBoost, not a neural network

This is one of the most important design decisions. Here's the full reasoning:

### The problem with neural networks for tabular financial data

Neural networks (LSTMs, Transformers, etc.) are powerful but have specific failure modes on this task:

| Problem | Detail |
|---------|--------|
| **Overfitting** | Financial time series are noisy. NNs have enormous capacity to memorize noise. |
| **Retraining cost** | A full LSTM/Transformer retrain takes hours, needs large GPU memory, and requires careful learning rate scheduling. Not viable daily. |
| **Feature importance opacity** | You can't easily ask "what drove this prediction?" — important for debugging bad calls. |
| **Non-stationarity sensitivity** | Market regimes shift. Deep models require large data distributions to generalize across regime changes. |
| **Warm-start complexity** | Continuing training on a neural network without catastrophic forgetting requires elastic weight consolidation, replay buffers, or fine-tuning tricks. XGBoost trees are additive and naturally composable. |

### Why XGBoost works well here

1. **Gradient boosting is inherently incremental.** Each tree corrects the residual error of all previous trees. Adding 30 new trees on yesterday's data is mathematically clean: you're fitting the gradient of the loss on the current residuals, conditioning on the full existing model.

2. **`tree_method="hist"` on GPU is extremely fast.** XGBoost's histogram-based algorithm bins continuous features into 256 buckets, making GPU acceleration highly efficient. Training 26 boosters on H100 takes ~20 minutes vs ~6 hours on CPU.

3. **Robust to irrelevant features.** Decision trees naturally ignore uninformative splits. You don't need to carefully regularize your feature set.

4. **`reg:squarederror` with log-returns is well-calibrated.** Log-returns are approximately normal, making MSE loss appropriate.

5. **Walk-forward validation is easy to implement correctly.** No gradient tape, no batch size to tune, no learning rate warmup.

6. **Interpretable.** Feature importance (gain, cover, weight) is built in. You can audit which features are driving predictions.

### Why not LightGBM or CatBoost?

The codebase has `train_lgbm.py` but it's not used in the live pipeline. LightGBM is comparable to XGBoost in accuracy for this problem. The reasons XGBoost was chosen:

- XGBoost's GPU backend (`hist` + `cuda`) has better VRAM efficiency on H100
- `xgb_model=init_model_path` provides clean warm-start: pass the existing booster, add trees on top
- XGBoost's Booster native format (`.json`) is more portable than LightGBM's binary

---

## 3. The 26-booster architecture

The central architectural decision: **one independent XGBoost booster per prediction horizon**.

### What this means

```
horizon_00.json  →  predicts log(close_09:30 / close_now)
horizon_01.json  →  predicts log(close_09:45 / close_now)
...
horizon_25.json  →  predicts log(close_15:45 / close_now)
```

26 models, each trained independently, each with its own trees, its own feature importances, its own hyperparameters.

### Why not one multi-output model?

| Approach | Pro | Con |
|----------|-----|-----|
| One model, 26 outputs | Simpler, shared feature learning | XGBoost doesn't natively handle multi-output regression with independent tree structure per target |
| 26 independent boosters | Each learns the structure of its specific horizon | 26× training time |
| Seq2Seq LSTM | Temporal structure between horizons | All neural network downsides above |

The independence is a feature, not a bug: the 09:30 bar has different predictability than the 15:45 bar. Forcing shared trees would average out horizon-specific signal. And on H100 with `ProcessPoolExecutor`, all 26 train in parallel anyway (see section 10).

### What "horizon" means statistically

Short horizons (h00, h01): highly autocorrelated, mean-reverting signal dominates. The 09:30 bar close is largely determined by the overnight gap and opening price.

Long horizons (h20–h25): more noise, harder to predict, market-wide momentum and macro signals matter more.

The model learns this automatically — the trees for short horizons will use different features with different splits than trees for long horizons.

---

## 4. Feature engineering

The feature set comes from two sources:

### A. Source columns (pre-computed in DB — `ml.market_data_15m`)

The database table has ~40 pre-computed features per row:

```
Price:      open, high, low, close, volume, slot_count
Lags:       lag_close_1, lag_close_5, lag_close_10
Diffs:      close_diff_1, close_diff_5
Returns:    pct_change_1, pct_change_5, log_return_1
SMAs:       sma_close_5, sma_close_10, sma_close_20, sma_volume_5
Time:       day_of_week, hour_of_day, day_monday..day_friday
            quarter_1..quarter_4, hour_early_morning..hour_late_afternoon
            month_of_year
Gap:        previous_close, overnight_gap_pct, overnight_log_return, is_gap_up, is_gap_down
```

### B. Derived market features (computed in `derive_market_features()`)

These are computed on-the-fly from the source columns:

```
Intraday context:
  bars_seen, bars_remaining         — how far into the day are we?
  intraday_progress                 — bars_seen / 26 (0.0 → 1.0)
  minutes_seen                      — bars_seen × 15

Log ratios (numerically stable):
  close_open_log_return             — log(close / open) for this bar
  high_low_log_range                — log(high / low) — intraday range
  close_vs_lag_close_1/5/10        — log(close / lag_close_N)
  close_vs_sma_close_5/10/20       — distance from SMA, in log space
  cutoff_return_from_open           — log(close / day_open)
  cutoff_price_vs_prev_close        — log(close / previous_close)

Volume (log-transformed to reduce skew):
  log_volume, log_slot_count
  log_cutoff_volume_seen            — cumulative volume today
  volume_vs_sma_volume_5            — current vol vs recent average

Running intraday stats:
  cutoff_high_seen, cutoff_low_seen — max/min price so far today
  cutoff_volume_seen                — cumulative volume
  cutoff_range_seen                 — log(high_seen / low_seen)
```

### C. Rolling cross-day features (the key signal)

The most powerful features are rolling statistics computed **per symbol, per slot_idx (same 15-min position across days)**:

```python
# For each (symbol, slot_idx) group — e.g. AAPL at 10:15:
# Rolling windows: [5, 10, 20] trading days

close_open_log_return_mean_5d    # average open→close return at this time of day
close_open_log_return_std_5d     # volatility of that return
cutoff_return_from_open_mean_10d  # how does today's intraday progress compare to the last 10?
overnight_gap_pct_mean_20d        # average gap for this symbol over 20 days
```

**Why is this powerful?** Markets have strong intraday seasonality. AAPL at 09:45 on Mondays behaves differently than AAPL at 14:30 on Fridays. By computing rolling stats per (symbol, slot_idx), the model learns "at this time of day, how does this stock typically move?" — essentially encoding intraday momentum and volatility regime in a simple rolling window.

### D. Winsorization

Extreme log-return values are clipped at the 0.5th and 99.5th percentile:

```python
winsor_pct = 0.005  # clip top/bottom 0.5%
```

Financial time series have heavy tails. A single extreme event (flash crash, earnings gap) can distort thousands of training examples. Winsorization prevents a few outlier rows from dominating tree splits.

**Why not log-transform volume instead of winsorize?** Volume is already log-transformed (`log_volume`). Winsorization applies to return-related features where the distribution is approximately symmetric around zero.

---

## 5. Target definition — why log-returns

```python
PRODUCTION_TARGET_DEFINITION = "log(next_regular_session_close_h / current_bar_close)"
```

In code:
```python
target_h{slot} = log( next_close_h{slot} / close )
```

### Why log, not raw return?

| Metric | Raw return | Log return |
|--------|-----------|------------|
| Distribution | Right-skewed (bounded at -100%) | Approximately symmetric, normal |
| Aggregation | Multiplicative | Additive |
| MSE loss | Penalizes large-price-stock moves more | Scale-invariant |

**Log-returns are additive**: if AAPL goes up 1% then down 0.5%, the compound return is `exp(log(1.01) + log(0.995)) - 1 ≈ 0.497%`. This makes them natural for multi-step prediction.

**Log-returns normalize across symbols**: A $0.01 move in AAPL ($200 stock) and NVDA ($900 stock) have the same log-return magnitude relative to their price. Training on raw prices would make the model think NVDA moves 4.5× more than AAPL.

**MSE on log-returns = approximately relative error**: minimizing `(log(pred/actual))^2` ≈ minimizing relative prediction error, which is what you care about in trading.

### Anomaly filtering

The code clips extreme targets:
```python
MAX_ABS_TARGET = 1.5  # log(4.48) — an ≈450% move; physically impossible in one day
```

Rows with `|target| > 1.5` are filtered out. These represent data errors, not real market events.

---

## 6. Warm-refresh — updating without forgetting

This is the core innovation of the system. Every day after market close, the model is updated with the day's new data **without retraining from scratch**.

### The XGBoost warm-start mechanism

```python
model.fit(X_new, y_new, xgb_model=init_model_path)
```

When you pass `xgb_model=path`, XGBoost:
1. Loads the existing booster (e.g. 1157 trees)
2. Computes the gradient of the loss on `X_new` **given the existing model's predictions**
3. Fits `warm_trees=30` new trees on that gradient
4. Appends them, giving you 1187 trees total

**The key insight**: each new tree is fitted on the *residual error* of the existing model on yesterday's data. The existing 1157 trees encode all historical signal — the new 30 trees specialize on recent patterns. No historical information is overwritten.

### Why 26 independent warm refreshes?

One refresh per horizon, per trading day:

```
Day 0: base model — horizon_00.json has 1157 trees
Day 1: warm refresh — horizon_00.json has 1187 trees (30 added)
Day 2: warm refresh — horizon_00.json has 1217 trees (30 more)
...
```

Each horizon is independently refreshed because the residual gradient at 09:30 is different from the residual gradient at 15:45. Mixing them would average out horizon-specific recent errors.

### The simulation (April 7, 2026)

The 8-hour NIBI job (job 12144848) is running a **full-day simulation** of warm-refresh on April 7:

```
Step 0:  base model predicts 09:30
         → compute residual
         → add 30 trees to horizon_00
         → refresh saves step_00/ bundle

Step 1:  step_00 model predicts 09:45 (base + 30 trees from 09:30)
         → add 30 trees to horizon_01
         → saves step_01/

...

Step 25: step_24 model predicts 15:45
         → writes SIMULATION_DONE sentinel
```

Each step simulates "what would the model have predicted at this point in the day if it had been continuously updated by intraday data?"

This is used to evaluate the warm-refresh strategy: does adding intraday bars actually improve prediction accuracy, or does it overfit noise?

---

## 7. Training pipeline — base model

The base model is built by `production_bootstrap()` in `XG_boost_3_multigpu_final.py`.

### Full pipeline

```
1. Load data
   parquet or DB → ml.market_data_15m
   Clip to train_end_date (avoids look-ahead)

2. Build dataset
   derive_market_features()          — intraday context features
   build_next_session_targets()      — next-day closes as targets
   add_group_rolling_features()      — cross-day rolling stats
   winsorize_columns()               — clip extremes

3. Restrict to training window
   Last 24 months by default         — avoids stale regime data

4. Walk-forward cross-validation
   evaluate_experiment() with n_folds=3, test_block_days=5
   Selects best hyperparameter variant

5. Final train on full window
   train_full_model_set()
   Trains all 26 horizon boosters
   Saves horizon_XX.json + model_manifest.json

6. Save artifacts
   metadata.json
   feature_names.json
   feature_importance_top25.csv
   run_summary.json
```

### Two training profiles

**Base profile** (`BEST_FIXED_VARIANT`):
- 24-month window of data
- 1157 trees, lr=0.015, max_depth=4
- Validated with expanding walk-forward CV
- This is the stable, long-memory model

**Fast-refresh profile** (`FAST_REFRESH_VARIANT`):
- Last 60 trading days only
- 400 trees, lr=0.03, max_depth=4
- Validated with last-block holdout (5 days)
- Faster to train, captures recent regime shifts
- Used as the starting point for intraday warm-refreshes

In production (`simulate_full_day.sbatch`), the base profile is used as the initial model, and fast-refresh parameters guide the warm-start updates.

---

## 8. Walk-forward validation

Walk-forward validation (also called "expanding window CV") is the correct way to validate time series models. Standard k-fold cross-validation is **wrong** for time series because it leaks future data into training.

### How it works

```
Timeline:  [──────────────── data ─────────────────]

Fold 1:    [train][test]
Fold 2:    [───train────][test]
Fold 3:    [──────train──────][test]
```

Each fold:
- Train on everything up to a cutoff date
- Test on the next `test_block_days=5` trading days
- Report RMSE, MAE, direction accuracy

**Direction accuracy** = fraction of predictions where `sign(predicted) == sign(actual)`. This is the metric that matters for trading: did the model correctly predict "up" or "down"?

### Why 3 folds?

`n_folds=3` with `test_block_days=5` means you're testing on 3 separate 5-day windows. This is a balance between:
- More folds = better estimate of generalization, but slower (26 models × 3 folds = 78 training runs)
- Fewer folds = faster, but noisier estimate

The training window is ~24 months × ~21 trading days = ~500 days. With 3 folds and 5-day test blocks, you're using the last ~250 days for validation.

### Exponential decay weighting

When computing sample weights:
```python
weights = 0.5 ^ (age_days / half_life_days)
```

Recent data gets higher weight. With `half_life=60`, data from 60 days ago gets weight 0.5, data from 120 days ago gets weight 0.25. This makes the model prioritize recent market regime over historical patterns.

---

## 9. Hyperparameters — chosen values and why

```python
BEST_FIXED_VARIANT = {
    "learning_rate": 0.015924834009065022,   # from Optuna
    "max_depth": 4,
    "subsample": 0.849153687610374,
    "colsample_bytree": 0.7446024807258872,
    "min_child_weight": 7.800340477474306,
    "n_estimators": 1157,                    # from Optuna
    "reg_alpha": 0.9916893895440203,         # L1
    "reg_lambda": 3.983536629736281,         # L2
}
```

These came from **Optuna hyperparameter search** (Bayesian optimization over the search space in `tune_params_for_final_horizon()`). Here's the reasoning behind each:

| Param | Value | Why |
|-------|-------|-----|
| `learning_rate` | 0.016 | Low LR + many trees. Lower LR = more trees needed, but generalizes better. |
| `max_depth` | 4 | Shallow trees. Prevents overfitting. Financial features rarely have depth-6 interactions. |
| `subsample` | 0.85 | Row sampling per tree. Reduces variance, like bagging. |
| `colsample_bytree` | 0.74 | Feature sampling per tree. Forces diversity in trees — some features appear only in some trees. |
| `min_child_weight` | 7.8 | Minimum sum of instance weights in a leaf. High value = fewer, safer splits. Prevents a tree from specializing on tiny subsets. |
| `n_estimators` | 1157 | Found by Optuna. Enough trees to capture signal without GPU OOM on H100. |
| `reg_alpha` | 0.99 | L1 regularization (Lasso-style). Drives small weights to zero — encourages sparsity. |
| `reg_lambda` | 3.98 | L2 regularization (Ridge-style). Smooths weight updates, prevents large individual feature impacts. |

### `max_depth=4` — the key regularization choice

Financial features have moderate interaction depth. The most important signals are:
- "Is the stock up more than X% from open AND volume is higher than average?"
- "Is it after 14:00 AND overnight gap was positive?"

These are depth-2 to depth-3 interactions. Going to depth 8 would start memorizing specific price patterns that don't generalize. Depth 4 is a sweet spot.

### Why not `n_estimators=2000` with even lower LR?

Memory. Each H100 can hold ~80GB VRAM. With 26 boosters × symbols × features, 1157 trees is near the efficient point where:
- Marginal improvement from more trees is < noise level
- Warm-start additions (30 trees/day) keep the model fresh without accumulating too many trees

---

## 10. Multi-GPU parallelism on H100

The H100 node at Alliance Canada (NIBI) has **1 H100 GPU with 80GB VRAM** per allocation (the `gpu` partition). The code was written to support multi-GPU (hence `_multigpu` in the filename), but the current allocation uses a single H100.

### How parallel training works

```python
num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
workers_per_gpu = int(os.environ.get("XGB_WORKERS_PER_GPU", "4"))
max_workers = num_gpus * workers_per_gpu  # 1 × 4 = 4 concurrent models
```

Each worker trains one horizon (booster). With 4 concurrent workers and 26 horizons, you need `ceil(26/4) = 7` rounds.

The key trick: `ProcessPoolExecutor` with a shared initializer that loads `X`, `y`, and params **once per worker** instead of serializing them 26 times (avoids ~GB of pickling overhead).

```python
with ProcessPoolExecutor(
    max_workers=max_workers,
    initializer=_gpu_worker_init,           # send X, y, params once
    initargs=(X, y, params, model_dir, ...),
) as executor:
    results = list(executor.map(_gpu_worker_train, range(26)))
```

Each worker sets `CUDA_VISIBLE_DEVICES = horizon_idx % num_gpus` to distribute across GPUs.

### Why not `n_jobs=-1` on XGBoost directly?

XGBoost's internal `n_jobs` uses OpenMP threads for CPU. On GPU, `n_jobs=-1` would cause multiple processes to compete for the same GPU — OOM or undefined behavior. Setting `n_jobs=1` in the GPU workers and parallelizing at the Python process level (via `ProcessPoolExecutor`) is the correct pattern.

---

## 11. Inference — how a prediction is made

At runtime, the backend calls `InferenceService.predict_stock_price()`:

```python
# 1. Load recent bars from DB (last N 15-min bars for this symbol)
recent_bars = crud.get_recent_inference_bars(session, symbol)

# 2. Run the same feature engineering as training
features = prepare_production_features(bars_df, model_bundle.feature_names)

# 3. Predict 26 log-returns, one per horizon booster
log_returns = model_bundle.predict(features)  # list of 26 floats

# 4. Convert to absolute prices
current_price = latest_bar["close"]
path = [current_price * exp(lr) for lr in log_returns]
```

**Critical**: the feature engineering at inference time must be **identical** to training time. This is enforced by `align_features_for_inference()` which:
1. Applies the same `derive_market_features()` pipeline
2. Drops all non-feature columns (targets, identifiers)
3. Reindexes to exactly `feature_names` — fills zeros for any missing (handles new/dropped features)

### Look-ahead prevention

The training cutoff (`train_end_date`) ensures the model never sees data from the simulation day. When predicting for April 7, the base model was trained through April 6.

At inference, `clip_bars_to_requested_as_of()` enforces this:
```python
# Only use bars with window_ts <= as_of_ts
# at 10:15 ET, only data through 10:15 is visible
bars = bars[bars["window_ts"] <= cutoff_ts]
```

This prevents the most common ML mistake in finance: accidentally including future prices in the feature row.

---

## 12. Model bundle on disk

The production model lives at `model_artifacts/current_base/` (atomic symlink).

### Directory structure

```
current_base/                       ← symlink → actual bundle dir
├── models/
│   ├── horizon_00.json             ← XGBoost native format booster #0
│   ├── horizon_01.json
│   ...
│   ├── horizon_25.json
│   └── model_manifest.json         ← maps target_h00 → filename
├── metadata.json                   ← training date, hyperparams, metrics
├── feature_names.json              ← ordered list of feature column names
├── feature_importance_top25.csv    ← gain-ranked feature importance
└── SIMULATION_DONE                 ← sentinel: written only when all 26 windows complete
```

### Why native JSON format, not pickle?

XGBoost's `.json` format is:
- **Portable**: readable in Python, R, C++, Java
- **Stable**: not tied to a specific sklearn version
- **Incrementally updatable**: `booster.load_model()` + `fit(..., xgb_model=path)` works cleanly

Pickle (`.pkl`) would fail if XGBoost is upgraded and the serialization format changes.

### Atomic symlink promotion

When a new model is promoted:
```python
tmp = symlink.with_suffix(".new")
tmp.symlink_to(new_bundle_path)
tmp.rename(symlink)        # atomic on Linux — kernel-level rename()
```

The backend loads `current_base` once at startup (`@lru_cache`). After promotion, new requests use the new model. In-flight predictions using the old cached bundle complete safely.

---

## 13. Key ML concepts explained

### Gradient Boosting

Build an ensemble of weak learners (trees) sequentially. Each new tree fits the **negative gradient of the loss function** on the current residuals — i.e., it tries to correct what the previous trees got wrong.

For MSE loss: gradient = predicted − actual. So each tree moves the prediction toward the true value by a small amount (`learning_rate`).

After N trees: `f(x) = f_0(x) + lr × t_1(x) + lr × t_2(x) + ... + lr × t_N(x)`

### Histogram-based algorithm (hist)

Instead of exact splits (which require sorting all N × features values), histogram-based XGBoost bins each feature into 256 buckets, builds a histogram, and finds the best split point in the histogram. This is:
- 10–100× faster than exact for large datasets
- Enables GPU acceleration (histograms are highly parallelizable)
- Slightly less precise splits, but the difference in accuracy is negligible

### Walk-forward validation vs k-fold

K-fold: shuffle data randomly, 80% train / 20% test.
→ **Wrong for time series**. If you train on Jan–Mar + May–Jul and test on Apr, you're training on data that came after the test set. The model "knows the future."

Walk-forward: train only on data before the test split.
→ Correct. Measures out-of-sample performance in the same way you'll use the model live.

### Overfitting and regularization

A model that memorizes training data and fails on new data. Signs: training RMSE << validation RMSE.

Regularization terms in XGBoost:
- `reg_lambda` (L2): penalizes large leaf weights → smooth predictions
- `reg_alpha` (L1): drives small weights to exactly zero → sparse models
- `subsample`: each tree sees only 85% of rows → reduces variance
- `colsample_bytree`: each tree sees only 74% of features → forces diversity
- `min_child_weight`: leaves need minimum effective weight → prevents rare-event overfitting

### Feature importance (gain)

For each feature, sum the gain (reduction in loss) it contributes across all splits in all 26 models. Higher gain = more useful feature.

In the model: `close_vs_sma_close_5`, `intraday_progress`, `overnight_gap_pct`, and rolling return features typically top the importance rankings.

### Log-return properties

```
Simple return:  r = (P1 - P0) / P0
Log return:     r = log(P1 / P0) = log(P1) - log(P0)
```

Properties:
- **Additive**: `log(P2/P0) = log(P2/P1) + log(P1/P0)` — multi-period returns just sum
- **Symmetric**: +10% followed by -10% = log(1.1) + log(0.9) = -0.005 (slightly negative, correctly)
- **Approximately normal**: for small returns, log(1+r) ≈ r, and financial returns are approximately normal
- **Scale-free**: a 1% move in a $10 stock and a $1000 stock have the same log-return

### Winsorization vs normalization

**Standardization** (subtract mean, divide by std): changes the scale but doesn't remove outliers.
**Winsorization**: clips values at percentile thresholds — removes extreme tail events.

For financial returns, winsorization is preferred because:
1. The tails are heavy — standardization doesn't help much
2. Extreme events are often data errors, not real signal
3. XGBoost trees are invariant to monotonic feature transformations — winsorization just removes the extreme splits that would only fit noise

---

## How it all connects — end-to-end flow

```
DB: ml.market_data_15m
    (8 symbols × 26 bars/day × ~500 days = ~100k rows)
         │
         ▼
XG_boost_3_multigpu_final.py
  derive_market_features()       — add intraday context
  build_next_session_targets()   — label each bar with next-day closes
  add_group_rolling_features()   — per-symbol, per-slot rolling stats
  winsorize_columns()
         │
         ▼
  production_bootstrap()
  train_full_model_set()         — 26 × XGBRegressor on H100
  [horizon_00.json ... horizon_25.json]
         │
         ▼
  simulate_full_day.sbatch (NIBI job)
  For each 15-min window 0..25:
    run_warm_refresh.py           — add 30 trees on intraday data
    Save step_XX/ bundle
    Write SIMULATION_DONE
         │
         ▼
  rsync artifacts → VPS
  model_artifacts/current_base/  (atomic symlink)
         │
         ▼
  FastAPI backend (inference/model_loader.py)
  NextDayPathBundle.predict()    — 26 boosters × 1 row = 26 log-returns
         │
         ▼
  Streamlit frontend
  Intraday price path chart
```

---

## Quick reference — key numbers

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `EXPECTED_REGULAR_BARS` | 26 | 15-min bars from 09:30–15:45 ET |
| `n_estimators` | 1157 | Base trees per horizon booster |
| `warm_trees` | 30 | Trees added per warm-refresh step |
| `learning_rate` | 0.016 | Step size for each new tree |
| `max_depth` | 4 | Max decision tree depth |
| `base_window_months` | 24 | Training history for base model |
| `fast_refresh_days` | 60 | Recent days for fast-refresh profile |
| `winsor_pct` | 0.005 | Clip top/bottom 0.5% of return features |
| `MAX_ABS_TARGET` | 1.5 | Max |log-return| = ~350% daily move (filter data errors) |
| Symbols | 8 | AAPL, AMD, AMZN, BA, MSFT, NVDA, GOOGL, TSLA |
