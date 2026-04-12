#!/usr/bin/env python3
"""
train_model.py
────────────────
Reads the Parquet snapshot produced by extract_and_save_parquet.py,
applies market-hours windowing, builds advanced features, runs Optuna HPO,
trains XGBoost, prunes weak features, and saves model + metrics.

Usage:
    python train_model.py --data_dir /scratch/$USER/ml/dt=2026-03-03 \
                          --model_out /scratch/$USER/ml/dt=2026-03-03/model

Requires (in virtualenv, with arrow module loaded beforehand on DRAC):
    pyarrow xgboost scikit-learn pandas numpy joblib optuna scipy
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ── Live log callback to VPS backend ─────────────────────────────────────────
VPS_LOG_URL  = os.getenv("VPS_LOG_URL", "http://cosc-vps:8000/api/v1/training")
_job_id   = os.environ.get("SLURM_JOB_ID", "local")
_run_date = time.strftime("%Y-%m-%d", time.gmtime())


def _post(path: str, payload: dict, timeout: int = 5):
    """Silent fire-and-forget POST — never crashes the training script."""
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{VPS_LOG_URL}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout)
    except Exception:
        pass  # never let logging kill training


def log(msg: str, level: str = "INFO", step: str = "", progress: float = -1):
    """Print locally AND stream to VPS frontend."""
    print(f"[{step or 'train'}] {msg}", flush=True)
    _post("/log", {
        "level":    level,
        "message":  msg,
        "step":     step,
        "progress": progress,
        "job_id":   _job_id,
        "run_date": _run_date,
    })


# ── Feature engineering ───────────────────────────────────────────────────────
def create_features(df_: pd.DataFrame, horizon: int = 78):
    df = df_.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    g  = df.groupby("ticker", group_keys=False)
    target_col = f"target_{horizon}"

    raw_fwd = g["close"].shift(-horizon) / df["close"] - 1.0
    df[target_col] = raw_fwd.ewm(span=3, min_periods=1).mean()  # light label smoothing

    for lag in [1, 3, 6, 12, 24, 48, 78]:
        df[f"ret_{lag}"] = g["close"].pct_change(lag)
    df["log_ret_1"] = np.log1p(df["ret_1"])

    df["hl_pct"]    = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_pct"]    = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["shadow_up"] = (df["high"] - df[["open","close"]].max(axis=1)) / df["close"].replace(0, np.nan)
    df["shadow_dn"] = (df[["open","close"]].min(axis=1) - df["low"]) / df["close"].replace(0, np.nan)

    for w in [12, 26, 52, 78, 156]:
        sma = g["close"].transform(lambda s: s.rolling(w).mean())
        ema = g["close"].transform(lambda s: s.ewm(span=w, adjust=False).mean())
        df[f"dist_sma_{w}"] = df["close"] / sma - 1.0
        df[f"dist_ema_{w}"] = df["close"] / ema - 1.0

    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = g["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_xover"]  = (df["macd_hist"] > 0).astype(np.int8)

    for w in [12, 36, 78, 156]:
        std = g["ret_1"].transform(lambda s: s.rolling(w).std())
        df[f"vol_{w}"] = std
        rs  = g["close"].transform(lambda s: s.rolling(w).std())
        rm  = g["close"].transform(lambda s: s.rolling(w).mean())
        df[f"bb_{w}_width"] = (2 * rs) / rm
        df[f"bb_{w}_pos"]   = (df["close"] - (rm - 2*rs)) / (4*rs + 1e-8)

    for rsi_w in [14, 28]:
        delta    = g["close"].diff()
        avg_gain = delta.clip(lower=0).groupby(df["ticker"]).transform(lambda s: s.rolling(rsi_w).mean())
        avg_loss = (-delta.clip(upper=0)).groupby(df["ticker"]).transform(lambda s: s.rolling(rsi_w).mean())
        df[f"rsi_{rsi_w}"]    = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
        df[f"rsi_{rsi_w}_os"] = (df[f"rsi_{rsi_w}"] < 30).astype(np.int8)
        df[f"rsi_{rsi_w}_ob"] = (df[f"rsi_{rsi_w}"] > 70).astype(np.int8)

    for w in [12, 36, 78]:
        vm = g["volume"].transform(lambda s: s.rolling(w).mean())
        df[f"vol_ratio_{w}"] = df["volume"] / (vm + 1)
    for w in [6, 12, 36]:
        df[f"vol_change_{w}"] = g["volume"].pct_change(w)
    df["vol_spike"] = (df["vol_ratio_12"] > 2).astype(np.int8)

    for w in [12, 36, 78]:
        df[f"dist_hi_{w}"] = df["close"] / g["high"].transform(lambda s: s.rolling(w).max()) - 1.0
        df[f"dist_lo_{w}"] = df["close"] / g["low"].transform(lambda s:  s.rolling(w).min()) - 1.0

    df["regime_vol"]      = g["ret_1"].transform(lambda s: s.rolling(78*90).std() * np.sqrt(78*252))
    df["high_vol_regime"] = (df["regime_vol"] > df["regime_vol"].median()).astype(np.int8)

    df["hour"]      = df["date"].dt.hour.astype(np.int8)
    df["minute"]    = df["date"].dt.minute.astype(np.int8)
    df["dow"]       = df["date"].dt.dayofweek.astype(np.int8)
    df["month"]     = df["date"].dt.month.astype(np.int8)
    df["is_monday"] = (df["dow"] == 0).astype(np.int8)
    df["is_friday"] = (df["dow"] == 4).astype(np.int8)
    df["is_open"]   = ((df["hour"] == 9) & (df["minute"] >= 30)).astype(np.int8)
    df["is_close"]  = (df["hour"] >= 15).astype(np.int8)

    enc = LabelEncoder()
    df["ticker_id"] = enc.fit_transform(df["ticker"].astype(str)).astype(np.int32)

    feature_cols = [c for c in df.columns if c not in {"ticker", "date", target_col}]
    dataset = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    mu, sig = dataset[target_col].mean(), dataset[target_col].std()
    dataset = dataset[dataset[target_col].between(mu - 3*sig, mu + 3*sig)].reset_index(drop=True)

    fc = dataset.select_dtypes(include=["float64"]).columns
    dataset[fc] = dataset[fc].astype(np.float32)

    return dataset, feature_cols, target_col, enc


def clean(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask].astype(np.float32), y[mask]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost from Parquet snapshot")
    parser.add_argument("--data_dir",     required=True, help="Directory containing *.parquet")
    parser.add_argument("--model_out",    required=True, help="Directory to save model artifacts")
    parser.add_argument("--horizon",      type=int, default=78,   help="Forward bars to predict (default=78=1day)")
    parser.add_argument("--optuna_trials",type=int, default=50,   help="Optuna HPO trials (0 to skip)")
    parser.add_argument("--device",       default="cuda",         help="cuda or cpu")
    parser.add_argument("--filter_market_hours", action="store_true", default=True)
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    model_out = Path(args.model_out)
    model_out.mkdir(parents=True, exist_ok=True)

    # ── Register job with VPS backend ────────────────────────────
    _post("/start", {"job_id": _job_id, "run_date": _run_date})
    log(f"Training job started — data_dir={args.data_dir}", level="SUCCESS", step="init")

    # ── 1. Load latest Parquet ─────────────────────────────────────
    parquet_files = sorted(data_dir.glob("*.parquet"), reverse=True)
    if not parquet_files:
        log(f"ERROR: No parquet files found in {data_dir}", level="ERROR", step="extract")
        sys.exit(1)

    log(f"Loading {parquet_files[0].name} ...", step="extract", progress=5)
    raw = pd.read_parquet(parquet_files[0])
    raw.columns = [c.lower() for c in raw.columns]
    if "symbol" in raw.columns and "ticker" not in raw.columns:
        raw = raw.rename(columns={"symbol": "ticker"})

    prices = raw[["ticker","date","open","high","low","close","volume"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker","date"]).reset_index(drop=True)
    log(f"{len(prices):,} rows loaded | {prices['ticker'].nunique()} tickers", step="extract", progress=10)

    # ── 2. Market-hours filter ────────────────────────────────────
    if args.filter_market_hours:
        h, m = prices["date"].dt.hour, prices["date"].dt.minute
        is_market = ((h > 9) | ((h == 9) & (m >= 30))) & (h < 16)
        prices = prices[is_market].reset_index(drop=True)
        log(f"After market-hours filter: {len(prices):,} rows", step="extract", progress=12)

    # ── 3. Feature engineering ────────────────────────────────────
    log("Building features ...", step="feature_eng", progress=15)
    t0 = time.time()
    dataset, feature_cols, target_col, encoder = create_features(prices, args.horizon)
    log(f"Dataset: {dataset.shape} | Features: {len(feature_cols)} ({time.time()-t0:.0f}s)",
        step="feature_eng", progress=30)

    # ── 4. Train/test split (temporal 80/20) ──────────────────────
    split_date = dataset["date"].quantile(0.80)
    train_df = dataset[dataset["date"] <= split_date]
    test_df  = dataset[dataset["date"] >  split_date]

    X_train, y_train = clean(train_df[feature_cols], train_df[target_col])
    X_test,  y_test  = clean(test_df[feature_cols],  test_df[target_col])
    log(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Split: {pd.Timestamp(split_date).date()}",
        step="feature_eng", progress=32)

    # ── 5. Optuna HPO ─────────────────────────────────────────────
    BEST_PARAMS = None
    if args.optuna_trials > 0:
        log(f"Starting Optuna HPO — {args.optuna_trials} trials ...", step="hpo", progress=35)
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            hpo_idx = np.random.default_rng(42).choice(len(X_train), size=min(200_000, len(X_train)), replace=False)
            Xh = X_train.iloc[hpo_idx]; yh = y_train.iloc[hpo_idx]
            sp  = int(0.8 * len(Xh))
            Xhtr, yhtr = Xh.iloc[:sp], yh.iloc[:sp]
            Xhva, yhva = Xh.iloc[sp:], yh.iloc[sp:]

            def objective(trial):
                p = {
                    "n_estimators"       : trial.suggest_int("n_estimators", 500, 3000, step=100),
                    "learning_rate"      : trial.suggest_float("learning_rate", 0.002, 0.05, log=True),
                    "max_depth"          : trial.suggest_int("max_depth", 4, 9),
                    "min_child_weight"   : trial.suggest_int("min_child_weight", 5, 50),
                    "subsample"          : trial.suggest_float("subsample", 0.5, 0.9),
                    "colsample_bytree"   : trial.suggest_float("colsample_bytree", 0.4, 0.9),
                    "colsample_bylevel"  : trial.suggest_float("colsample_bylevel", 0.4, 0.9),
                    "reg_alpha"          : trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                    "reg_lambda"         : trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
                    "gamma"              : trial.suggest_float("gamma", 0.0, 5.0),
                    "tree_method"        : "hist",
                    "device"             : args.device,
                    "objective"          : "reg:squarederror",
                    "eval_metric"        : "rmse",
                    "random_state"       : 42,
                    "early_stopping_rounds": 30,
                }
                m = XGBRegressor(**p)
                m.fit(Xhtr, yhtr, eval_set=[(Xhva, yhva)], verbose=False)
                return np.sqrt(mean_squared_error(yhva, m.predict(Xhva)))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=True)
            BEST_PARAMS = study.best_params
            log(f"HPO complete — best RMSE: {study.best_value:.6f}", level="SUCCESS", step="hpo", progress=55)
            log(f"Best params: {json.dumps(BEST_PARAMS)}", step="hpo")
        except ImportError:
            log("optuna not installed — using default params", level="WARN", step="hpo")

    # ── 6. Final model training ───────────────────────────────────
    params = BEST_PARAMS or {}
    params.update({
        "n_estimators"         : params.get("n_estimators", 5000),
        "learning_rate"        : params.get("learning_rate", 0.004),
        "max_depth"            : params.get("max_depth", 6),
        "min_child_weight"     : params.get("min_child_weight", 25),
        "subsample"            : params.get("subsample", 0.75),
        "colsample_bytree"     : params.get("colsample_bytree", 0.65),
        "colsample_bylevel"    : params.get("colsample_bylevel", 0.65),
        "reg_alpha"            : params.get("reg_alpha", 1.5),
        "reg_lambda"           : params.get("reg_lambda", 6.0),
        "gamma"                : params.get("gamma", 0.5),
        "tree_method"          : "hist",
        "device"               : args.device,
        "objective"            : "reg:squarederror",
        "eval_metric"          : "rmse",
        "random_state"         : 42,
        "early_stopping_rounds": 100,
    })

    log("Training final XGBoost model ...", step="train", progress=58)
    t1 = time.time()
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=200)
    train_time = time.time() - t1

    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae  = float(mean_absolute_error(y_test, pred))
    da   = float(np.mean(np.sign(pred) == np.sign(y_test)))
    log(f"RMSE={rmse:.6f}  MAE={mae:.6f}  DirAcc={da:.4f}  ({train_time:.0f}s)",
        level="SUCCESS", step="train", progress=80)

    # ── 7. Feature pruning ────────────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    selected   = importance[importance >= 0.0001].sort_values(ascending=False).index.tolist()
    dropped    = importance[importance < 0.0001].index.tolist()
    log(f"Pruning {len(dropped)} weak features → keeping {len(selected)}", step="train", progress=82)

    X_tr2, y_tr2 = clean(train_df[selected], train_df[target_col])
    X_te2, y_te2 = clean(test_df[selected],  test_df[target_col])
    model2 = XGBRegressor(**params)
    model2.fit(X_tr2, y_tr2, eval_set=[(X_te2, y_te2)], verbose=False)
    pred2  = model2.predict(X_te2)
    rmse2  = float(np.sqrt(mean_squared_error(y_te2, pred2)))
    da2    = float(np.mean(np.sign(pred2) == np.sign(y_te2)))

    if rmse2 <= rmse:
        log(f"Pruned model better ({rmse2:.6f} < {rmse:.6f}) — using it", level="SUCCESS", step="train", progress=88)
        model, feature_cols, pred = model2, selected, pred2
        rmse, da = rmse2, da2
        X_test, y_test = X_te2, y_te2
    else:
        log(f"Original model better — keeping it (RMSE={rmse:.6f})", step="train", progress=88)

    # ── 8. Save artifacts ─────────────────────────────────────────
    joblib.dump(model,   model_out / "model.pkl")
    joblib.dump(encoder, model_out / "encoder.pkl")

    metrics = {
        "rmse"            : rmse,
        "mae"             : mae,
        "dir_accuracy"    : da,
        "best_iteration"  : int(model.best_iteration),
        "n_features"      : len(feature_cols),
        "train_rows"      : len(X_train),
        "test_rows"       : len(X_test),
        "split_date"      : str(pd.Timestamp(split_date).date()),
        "horizon_bars"    : args.horizon,
        "train_time_sec"  : round(train_time, 1),
        "feature_cols"    : feature_cols,
        "tickers"         : sorted(prices["ticker"].unique().tolist()),
        "hpo_params"      : BEST_PARAMS,
    }
    with open(model_out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"Artifacts saved to {model_out}", step="save", progress=95)
    log(f"DONE — RMSE={rmse:.6f}  DirAcc={da:.4f}  Features={len(feature_cols)}",
        level="SUCCESS", step="done", progress=100)


if __name__ == "__main__":
    main()
