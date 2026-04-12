"""
Feature engineering for stock prediction inference.

Calculates the exact feature set required by the global XGBoost model.
"""

from typing import Any

import numpy as np
import pandas as pd

EXPECTED_REGULAR_BARS = 26


def calculate_technical_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build the full feature frame for one ticker."""
    frame = df.copy()
    frame["ticker"] = symbol
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = frame[col].astype(float)

    grouped = frame.groupby("ticker", group_keys=False)

    frame["ret_1"] = grouped["close"].pct_change(1)
    frame["ret_5"] = grouped["close"].pct_change(5)
    frame["ret_10"] = grouped["close"].pct_change(10)
    frame["ret_20"] = grouped["close"].pct_change(20)
    frame["log_ret_1"] = np.log1p(frame["ret_1"])
    frame["hl_spread"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
    frame["oc_spread"] = (frame["close"] - frame["open"]) / frame["open"].replace(0, np.nan)

    for window in [10, 20, 50, 100]:
        frame[f"sma_{window}"] = grouped["close"].transform(lambda s: s.rolling(window).mean())
        frame[f"ema_{window}"] = grouped["close"].transform(
            lambda s: s.ewm(span=window, adjust=False).mean()
        )
        frame[f"dist_sma_{window}"] = frame["close"] / frame[f"sma_{window}"] - 1.0
        frame[f"dist_ema_{window}"] = frame["close"] / frame[f"ema_{window}"] - 1.0

    frame["ema_12"] = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    frame["ema_26"] = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    frame["macd"] = frame["ema_12"] - frame["ema_26"]
    frame["macd_signal"] = grouped["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]

    for window in [10, 20, 60]:
        frame[f"volatility_{window}"] = grouped["ret_1"].transform(lambda s: s.rolling(window).std())
        roll_std = grouped["close"].transform(lambda s: s.rolling(window).std())
        roll_mean = grouped["close"].transform(lambda s: s.rolling(window).mean())
        frame[f"bb_{window}_width"] = (4.0 * roll_std) / roll_mean

    delta = grouped["close"].diff()
    avg_gain = delta.clip(lower=0).groupby(frame["ticker"]).transform(lambda s: s.rolling(14).mean())
    avg_loss = (-delta.clip(upper=0)).groupby(frame["ticker"]).transform(lambda s: s.rolling(14).mean())
    frame["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    frame["vol_ma_20"] = grouped["volume"].transform(lambda s: s.rolling(20).mean())
    frame["vol_ratio_20"] = frame["volume"] / frame["vol_ma_20"]

    for window in [5, 20, 60]:
        frame[f"vol_change_{window}"] = grouped["volume"].pct_change(window)

    frame["rolling_max_10"] = grouped["high"].transform(lambda s: s.rolling(10).max())
    frame["rolling_min_10"] = grouped["low"].transform(lambda s: s.rolling(10).min())
    frame["dist_roll_max_10"] = frame["close"] / frame["rolling_max_10"] - 1.0
    frame["dist_roll_min_10"] = frame["close"] / frame["rolling_min_10"] - 1.0
    frame["hl_pct"] = (frame["high"] - frame["low"]) / frame["close"] * 100

    return frame


def prepare_features_for_prediction(
    df: pd.DataFrame,
    symbol: str,
    ticker_encoder: Any,
    feature_cols_order: list[str],
) -> pd.DataFrame:
    """Prepare one latest-row feature vector for the prediction model."""
    df_with_features = calculate_technical_indicators(df, symbol)

    try:
        ticker_id = ticker_encoder.transform([symbol])[0]
    except ValueError:
        ticker_id = 0

    df_with_features["ticker_id"] = ticker_id
    latest_features = df_with_features.iloc[[-1]].copy().fillna(0)
    return latest_features[feature_cols_order].astype("float32")


def _build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 514 next-day model features from a DataFrame of raw 15-min bars.

    Input columns required: date, open, high, low, close, volume, trade_count, vwap
    Returns one row per complete trading day, indexed by trading_date.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["date"])
    df = df.sort_values("ts").reset_index(drop=True)

    # Trading date = calendar date of the bar
    df["trading_date"] = df["ts"].dt.normalize()
    total_min = df["ts"].dt.hour * 60 + df["ts"].dt.minute

    # Regular session: 09:30 (570) to 15:45 (945) inclusive
    df["is_regular"] = (total_min >= 570) & (total_min <= 945)
    # Premarket: anything before 09:30
    df["is_premarket"] = total_min < 570
    # Slot index within regular session (0=09:30, 25=15:45)
    df["slot"] = np.where(df["is_regular"], (total_min - 570) // 15, -1)

    trading_dates = sorted(df["trading_date"].unique())
    daily_rows: list[dict] = []
    prev_close: float = float("nan")

    for td in trading_dates:
        day_df = df[df["trading_date"] == td]
        reg = day_df[day_df["is_regular"]].sort_values("ts")
        pm = day_df[day_df["is_premarket"]].sort_values("ts")

        # Skip days with no regular session data
        if len(reg) == 0:
            continue

        # --- Regular session aggregates ---
        day_open = float(reg["open"].iloc[0])
        day_close = float(reg["close"].iloc[-1])
        day_volume = float(reg["volume"].sum())
        day_trades = float(reg["trade_count"].sum())
        day_high = float(reg["high"].max())
        day_low = float(reg["low"].min())

        closes = reg["close"].values.astype(float)
        prev_closes = np.concatenate([[day_open], closes[:-1]])
        log_rets = np.log(np.where(prev_closes > 0, closes / prev_closes, 1.0))
        day_realized_vol = float(log_rets.std() * np.sqrt(26)) if len(log_rets) > 1 else 0.0

        full_day_return = (day_close - day_open) / day_open if day_open > 0 else 0.0
        day_range = (day_high - day_low) / day_open if day_open > 0 else 0.0

        vwap_num = float((reg["vwap"] * reg["volume"]).sum())
        day_vwap = vwap_num / day_volume if day_volume > 0 else day_open
        day_vwap_delta = (day_vwap - day_open) / day_open if day_open > 0 else 0.0

        overnight_gap = (day_open - prev_close) / prev_close if np.isfinite(prev_close) and prev_close > 0 else 0.0

        # --- Premarket aggregates ---
        if len(pm) > 0:
            has_premarket = 1.0
            pm_open = float(pm["open"].iloc[0])
            pm_close = float(pm["close"].iloc[-1])
            pm_volume = float(pm["volume"].sum())
            pm_trades = float(pm["trade_count"].sum())
            pm_high = float(pm["high"].max())
            pm_low = float(pm["low"].min())

            pm_closes = pm["close"].values.astype(float)
            pm_prev = np.concatenate([[pm_open], pm_closes[:-1]])
            pm_log_rets = np.log(np.where(pm_prev > 0, pm_closes / pm_prev, 1.0))
            premarket_realized_vol = float(pm_log_rets.std()) if len(pm_log_rets) > 1 else 0.0

            premarket_return = (pm_close - pm_open) / pm_open if pm_open > 0 else 0.0
            premarket_range = (pm_high - pm_low) / pm_open if pm_open > 0 else 0.0
            premarket_gap = (pm_open - prev_close) / prev_close if np.isfinite(prev_close) and prev_close > 0 else 0.0
            premarket_close_to_prev = (pm_close - prev_close) / prev_close if np.isfinite(prev_close) and prev_close > 0 else 0.0

            pm_vwap_num = float((pm["vwap"] * pm["volume"]).sum())
            pm_vwap = pm_vwap_num / pm_volume if pm_volume > 0 else pm_open
            premarket_vwap_delta = (pm_vwap - prev_close) / prev_close if np.isfinite(prev_close) and prev_close > 0 else 0.0
        else:
            has_premarket = pm_volume = pm_trades = 0.0
            premarket_realized_vol = premarket_return = premarket_range = 0.0
            premarket_gap = premarket_close_to_prev = premarket_vwap_delta = 0.0

        # --- Per-slot features ---
        slot_feats: dict = {}
        for s in range(26):
            pfx = f"slot_{s:02d}"
            sb = reg[reg["slot"] == s]
            if len(sb) > 0:
                bar = sb.iloc[0]
                bc = float(bar["close"])
                bh = float(bar["high"])
                bl = float(bar["low"])
                bv = float(bar["volume"])
                if s == 0:
                    pbc = day_open
                else:
                    prev_sb = reg[reg["slot"] == s - 1]
                    pbc = float(prev_sb["close"].iloc[-1]) if len(prev_sb) > 0 else day_open
                bar_ret = float(np.log(bc / pbc)) if pbc > 0 else 0.0
                path_from_open = (bc - day_open) / day_open if day_open > 0 else 0.0
                path_to_close = (day_close - bc) / day_open if day_open > 0 else 0.0
                slot_range = (bh - bl) / day_open if day_open > 0 else 0.0
                vol_share = bv / day_volume if day_volume > 0 else 0.0
            else:
                bar_ret = path_from_open = path_to_close = slot_range = vol_share = 0.0

            slot_feats[f"{pfx}_slot_bar_return"] = bar_ret
            slot_feats[f"{pfx}_slot_path_from_open"] = path_from_open
            slot_feats[f"{pfx}_slot_path_to_close"] = path_to_close
            slot_feats[f"{pfx}_slot_range"] = slot_range
            slot_feats[f"{pfx}_slot_volume_share"] = vol_share

        prev_close = day_close

        daily_rows.append({
            "trading_date": td,
            "day_volume": day_volume,
            "day_trades": day_trades,
            "day_realized_vol": day_realized_vol,
            "full_day_return": full_day_return,
            "day_range": day_range,
            "overnight_gap": overnight_gap,
            "day_vwap_delta": day_vwap_delta,
            "premarket_volume": pm_volume,
            "premarket_trades": pm_trades,
            "premarket_realized_vol": premarket_realized_vol,
            "premarket_return": premarket_return,
            "premarket_range": premarket_range,
            "premarket_gap": premarket_gap,
            "premarket_close_to_prev_close": premarket_close_to_prev,
            "premarket_vwap_delta": premarket_vwap_delta,
            "has_premarket": has_premarket,
            **slot_feats,
        })

    daily_df = pd.DataFrame(daily_rows).set_index("trading_date")

    # --- Rolling windows ---
    daily_cols = [
        "full_day_return", "day_range", "day_volume", "day_realized_vol",
        "overnight_gap", "premarket_return", "premarket_volume",
    ]
    for window, suffix in [(5, "5d"), (10, "10d"), (20, "20d"), (60, "60d")]:
        for col in daily_cols:
            daily_df[f"{col}_mean_{suffix}"] = daily_df[col].rolling(window, min_periods=1).mean()
            daily_df[f"{col}_std_{suffix}"] = daily_df[col].rolling(window, min_periods=1).std().fillna(0.0)

    for window, suffix in [(5, "5d"), (10, "10d"), (20, "20d")]:
        for s in range(26):
            pfx = f"slot_{s:02d}"
            for metric in ["slot_path_from_open", "slot_volume_share"]:
                col = f"{pfx}_{metric}"
                daily_df[f"{col}_mean_{suffix}"] = daily_df[col].rolling(window, min_periods=1).mean()
                daily_df[f"{col}_std_{suffix}"] = daily_df[col].rolling(window, min_periods=1).std().fillna(0.0)

    return daily_df


def prepare_features_for_next_day(
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build all 514 next-day model features from raw 15-min bar data.

    Args:
        df: Raw 15-min bars with columns: date, open, high, low, close, volume,
            trade_count, vwap.  Must span at least 65 trading days for accurate
            60d rolling features.
        feature_names: Ordered list of 514 feature names from feature_names.json.

    Returns:
        Single-row DataFrame (1, 514) ready for xgboost.DMatrix, float32.

    Raises:
        KeyError: If a required feature cannot be computed from the input data.
    """
    daily_df = _build_daily_features(df)

    if len(daily_df) == 0:
        raise ValueError("No complete trading days found in the provided bar data.")

    last_row = daily_df.iloc[[-1]].reset_index(drop=True)

    missing = [f for f in feature_names if f not in last_row.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} required features for next-day model: "
            f"{missing[:10]}{'…' if len(missing) > 10 else ''}"
        )

    return last_row[feature_names].fillna(0.0).astype("float32")


def safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = pd.to_numeric(numerator, errors="coerce").astype(float) / pd.to_numeric(
        denominator, errors="coerce"
    ).astype(float)
    ratio = ratio.where(ratio > 0)
    return np.log(ratio).replace([np.inf, -np.inf], np.nan)


def add_group_rolling_features_by_keys(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    windows: list[int],
    suffix_template: str,
) -> pd.DataFrame:
    df = df.sort_values(group_cols + ["trade_date"]).copy()
    grouped = df.groupby(group_cols, observed=True)
    feature_map: dict[str, pd.Series] = {}
    for window in windows:
        for col in value_cols:
            feature_map[suffix_template.format(col=col, window=window, stat="mean")] = grouped[col].transform(
                lambda s: s.rolling(window, min_periods=2).mean()
            )
            feature_map[suffix_template.format(col=col, window=window, stat="std")] = grouped[col].transform(
                lambda s: s.rolling(window, min_periods=2).std(ddof=0)
            )
    if not feature_map:
        return df
    return pd.concat([df, pd.DataFrame(feature_map, index=df.index)], axis=1).copy()


def derive_market_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the production inference features from engineered `ml.market_data_15m`
    history, matching the training pipeline for the active 176-feature bundle.
    """
    out = bars.sort_values(["symbol", "window_ts"]).copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out["slot_idx"] = out.groupby(["symbol", "trade_date"], observed=True).cumcount()
    out["bar_time"] = out["window_ts"].dt.strftime("%H:%M")

    day_group = out.groupby(["symbol", "trade_date"], observed=True)
    day_open = day_group["open"].transform("first")

    out["bars_seen"] = out["slot_idx"].astype(int) + 1
    out["bars_remaining"] = EXPECTED_REGULAR_BARS - out["bars_seen"]
    out["minutes_seen"] = out["bars_seen"].astype(float) * 15.0
    out["intraday_progress"] = out["bars_seen"].astype(float) / float(EXPECTED_REGULAR_BARS)

    out["cutoff_close"] = pd.to_numeric(out["close"], errors="coerce").astype(float)
    out["cutoff_high_seen"] = day_group["high"].cummax().astype(float)
    out["cutoff_low_seen"] = day_group["low"].cummin().astype(float)
    out["cutoff_volume_seen"] = day_group["volume"].cumsum().astype(float)
    out["cutoff_slot_idx"] = out["slot_idx"].astype(int)

    out["close_open_log_return"] = safe_log_ratio(out["close"], out["open"])
    out["high_low_log_range"] = safe_log_ratio(out["high"], out["low"])
    out["close_vs_lag_close_1"] = safe_log_ratio(out["close"], out["lag_close_1"])
    out["close_vs_lag_close_5"] = safe_log_ratio(out["close"], out["lag_close_5"])
    out["close_vs_lag_close_10"] = safe_log_ratio(out["close"], out["lag_close_10"])
    out["close_vs_sma_close_5"] = safe_log_ratio(out["close"], out["sma_close_5"])
    out["close_vs_sma_close_10"] = safe_log_ratio(out["close"], out["sma_close_10"])
    out["close_vs_sma_close_20"] = safe_log_ratio(out["close"], out["sma_close_20"])
    out["cutoff_return_from_open"] = safe_log_ratio(out["close"], day_open)
    out["cutoff_range_seen"] = safe_log_ratio(out["cutoff_high_seen"], out["cutoff_low_seen"])
    out["cutoff_price_vs_prev_close"] = safe_log_ratio(out["close"], out["previous_close"])
    out["log_volume"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0).fillna(0.0))
    out["log_slot_count"] = np.log1p(
        pd.to_numeric(out["slot_count"], errors="coerce").clip(lower=0).fillna(0.0)
    )
    out["log_cutoff_volume_seen"] = np.log1p(out["cutoff_volume_seen"].clip(lower=0).fillna(0.0))
    out["volume_vs_sma_volume_5"] = (
        np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0).fillna(0.0))
        - np.log1p(pd.to_numeric(out["sma_volume_5"], errors="coerce").clip(lower=0).fillna(0.0))
    )
    return out


def prepare_production_features(
    bars: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build the active production model's 176-feature vector from recent
    `ml.market_data_15m` history for a single symbol.
    """
    dataset = derive_market_features(bars)

    roll_cols = [
        "close_open_log_return",
        "high_low_log_range",
        "pct_change_1",
        "pct_change_5",
        "log_return_1",
        "cutoff_return_from_open",
        "cutoff_range_seen",
        "cutoff_price_vs_prev_close",
        "cutoff_volume_seen",
        "overnight_gap_pct",
        "overnight_log_return",
    ]
    dataset = add_group_rolling_features_by_keys(
        dataset,
        group_cols=["symbol", "slot_idx"],
        value_cols=[col for col in roll_cols if col in dataset.columns],
        windows=[5, 10, 20, 60],
        suffix_template="{col}_{stat}_{window}d",
    )

    short_roll_cols = [
        "cutoff_return_from_open",
        "cutoff_range_seen",
        "cutoff_volume_seen",
        "pct_change_1",
    ]
    dataset = add_group_rolling_features_by_keys(
        dataset,
        group_cols=["symbol", "slot_idx"],
        value_cols=[col for col in short_roll_cols if col in dataset.columns],
        windows=[5, 10, 20],
        suffix_template="{col}_{stat}_{window}cut",
    )

    if dataset.empty:
        raise ValueError("No engineered market rows available for production inference.")

    last_row = dataset.sort_values("window_ts").iloc[[-1]].copy()
    for col in feature_names:
        if col not in last_row.columns:
            last_row[col] = 0.0
    return last_row[feature_names].fillna(0.0).astype("float32")
