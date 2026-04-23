#!/usr/bin/env python3
"""
Generate report-ready prediction, accuracy, and loss tables from simulation artifacts.

The model predicts the next regular session path. For forecast origins on the
simulation date, accuracy is measured against the following trading date.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_DIR = REPO_ROOT / "model_artifacts" / "current_simulation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "model_submission_tables"
PSQL_COLUMNS = ["symbol", "trade_date", "window_ts", "close"]
HORIZON_COUNT = 26


@dataclass(frozen=True)
class ReportInputs:
    simulation_dir: Path
    output_dir: Path
    db_container: str
    db_user: str
    db_name: str
    forecast_date: str | None
    truth_date: str | None


def parse_args() -> ReportInputs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-container", default="algo-trade-monorepo-db-1")
    parser.add_argument("--db-user", default="appuser")
    parser.add_argument("--db-name", default="algotrade")
    parser.add_argument("--forecast-date", help="Forecast-origin date, e.g. 2026-04-07")
    parser.add_argument("--truth-date", help="Next-session truth date, e.g. 2026-04-08")
    args = parser.parse_args()

    return ReportInputs(
        simulation_dir=args.simulation_dir,
        output_dir=args.output_dir,
        db_container=args.db_container,
        db_user=args.db_user,
        db_name=args.db_name,
        forecast_date=args.forecast_date,
        truth_date=args.truth_date,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_dates(simulation_dir: Path, forecast_date: str | None, truth_date: str | None) -> tuple[str, str]:
    if forecast_date and truth_date:
        return forecast_date, truth_date

    summary_path = simulation_dir / "simulation_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    inferred_forecast = forecast_date or summary.get("replay_date")
    if not inferred_forecast:
        first_csv = simulation_dir / "step_00" / "predictions" / "predictions.csv"
        first = pd.read_csv(first_csv, nrows=1)
        inferred_forecast = str(pd.to_datetime(first["as_of_ts"].iloc[0]).date())

    if truth_date:
        return inferred_forecast, truth_date

    forecast_ts = pd.Timestamp(inferred_forecast)
    truth_ts = forecast_ts + pd.offsets.BDay(1)
    return inferred_forecast, str(truth_ts.date())


def load_predictions(simulation_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for step in range(HORIZON_COUNT):
        csv_path = simulation_dir / f"step_{step:02d}" / "predictions" / "predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "step" not in frame.columns:
            frame["step"] = step
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_actual_bars(inputs: ReportInputs, forecast_date: str, truth_date: str) -> pd.DataFrame:
    query = f"""
COPY (
    SELECT
        symbol,
        trade_date::text AS trade_date,
        window_ts,
        close::float AS close
    FROM ml.market_data_15m
    WHERE trade_date IN ('{forecast_date}', '{truth_date}')
      AND window_ts AT TIME ZONE 'America/New_York' >= trade_date + TIME '09:30'
      AND window_ts AT TIME ZONE 'America/New_York' <= trade_date + TIME '15:45'
    ORDER BY symbol, window_ts
) TO STDOUT WITH CSV HEADER
"""
    cmd = [
        "docker",
        "exec",
        "-i",
        inputs.db_container,
        "psql",
        "-U",
        inputs.db_user,
        "-d",
        inputs.db_name,
        "-c",
        query,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) <= 1:
        raise RuntimeError(f"No actual bars returned for {forecast_date} / {truth_date}")

    from io import StringIO

    bars = pd.read_csv(StringIO("\n".join(rows)))
    bars["window_ts"] = pd.to_datetime(bars["window_ts"], utc=True)
    bars["trade_date"] = bars["trade_date"].astype(str)
    return bars[PSQL_COLUMNS]


def direction_label(values: pd.Series) -> pd.Series:
    return np.where(values.astype(float) > 0, "up", "down")


def build_actual_wide(bars: pd.DataFrame, date: str, prefix: str) -> pd.DataFrame:
    day = bars[bars["trade_date"] == date].sort_values(["symbol", "window_ts"]).copy()
    day["bar_idx"] = day.groupby("symbol").cumcount()
    complete_symbols = day.groupby("symbol")["bar_idx"].nunique()
    keep = complete_symbols[complete_symbols == HORIZON_COUNT].index
    day = day[day["symbol"].isin(keep)]

    wide = day.pivot(index="symbol", columns="bar_idx", values="close").reindex(columns=range(HORIZON_COUNT))
    wide.columns = [f"{prefix}_close_h{idx:02d}" for idx in range(HORIZON_COUNT)]
    return wide.reset_index()


def build_scored_tables(predictions: pd.DataFrame, bars: pd.DataFrame, forecast_date: str, truth_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = bars[bars["trade_date"] == forecast_date][["symbol", "window_ts", "close"]].rename(
        columns={"window_ts": "as_of_ts", "close": "cutoff_close"}
    )
    forecast["as_of_ts"] = pd.to_datetime(forecast["as_of_ts"], utc=True)

    truth_wide = build_actual_wide(bars, truth_date, "actual")
    scored = predictions.copy()
    scored["as_of_ts"] = pd.to_datetime(scored["as_of_ts"], utc=True)
    scored = scored.merge(forecast, on=["symbol", "as_of_ts"], how="inner")
    scored = scored.merge(truth_wide, on="symbol", how="inner")

    pred_cols = [f"pred_log_return_h{idx:02d}" for idx in range(HORIZON_COUNT)]
    actual_cols = [f"actual_close_h{idx:02d}" for idx in range(HORIZON_COUNT)]
    for idx in range(HORIZON_COUNT):
        scored[f"actual_log_return_h{idx:02d}"] = np.log(scored[f"actual_close_h{idx:02d}"] / scored["cutoff_close"])
        scored[f"pred_close_h{idx:02d}"] = scored["cutoff_close"] * np.exp(scored[f"pred_log_return_h{idx:02d}"])

    scored["actual_final_close"] = scored["actual_close_h25"]
    scored["predicted_final_close"] = scored["pred_close_h25"]
    scored["actual_final_log_return"] = scored["actual_log_return_h25"]
    scored["predicted_final_log_return"] = scored["pred_log_return_h25"]
    scored["actual_final_return_pct"] = (np.exp(scored["actual_final_log_return"]) - 1.0) * 100.0
    scored["predicted_final_return_pct"] = (np.exp(scored["predicted_final_log_return"]) - 1.0) * 100.0
    scored["actual_direction"] = direction_label(scored["actual_final_log_return"])
    scored["correct_direction"] = scored["actual_direction"] == scored["predicted_direction"]
    scored["final_log_error"] = scored["predicted_final_log_return"] - scored["actual_final_log_return"]
    scored["final_abs_log_error"] = scored["final_log_error"].abs()
    scored["final_squared_log_error"] = scored["final_log_error"] ** 2
    scored["final_price_error"] = scored["predicted_final_close"] - scored["actual_final_close"]
    scored["final_abs_price_error"] = scored["final_price_error"].abs()

    long_frames: list[pd.DataFrame] = []
    for idx in range(HORIZON_COUNT):
        long_frames.append(
            pd.DataFrame(
                {
                    "symbol": scored["symbol"],
                    "step": scored["step"],
                    "slot_label": scored["slot_label"],
                    "horizon": idx,
                    "pred_log_return": scored[pred_cols[idx]],
                    "actual_log_return": scored[f"actual_log_return_h{idx:02d}"],
                    "pred_close": scored[f"pred_close_h{idx:02d}"],
                    "actual_close": scored[actual_cols[idx]],
                }
            )
        )
    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["log_error"] = long_df["pred_log_return"] - long_df["actual_log_return"]
    long_df["abs_log_error"] = long_df["log_error"].abs()
    long_df["squared_log_error"] = long_df["log_error"] ** 2
    long_df["price_error"] = long_df["pred_close"] - long_df["actual_close"]
    long_df["abs_price_error"] = long_df["price_error"].abs()
    return scored, long_df


def aggregate_step_metrics(scored: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    path_metrics = long_df.groupby("step").agg(
        path_mse_loss=("squared_log_error", "mean"),
        path_rmse=("squared_log_error", lambda s: math.sqrt(float(s.mean()))),
        path_mae=("abs_log_error", "mean"),
        price_path_rmse=("price_error", lambda s: math.sqrt(float(np.mean(np.square(s))))),
        price_path_mae=("abs_price_error", "mean"),
    )
    final_metrics = scored.groupby("step").agg(
        rows=("symbol", "count"),
        slot_label=("slot_label", "first"),
        as_of_ts=("as_of_ts", "first"),
        direction_accuracy=("correct_direction", "mean"),
        final_mse_loss=("final_squared_log_error", "mean"),
        final_horizon_rmse=("final_squared_log_error", lambda s: math.sqrt(float(s.mean()))),
        final_horizon_mae=("final_abs_log_error", "mean"),
        final_price_rmse=("final_price_error", lambda s: math.sqrt(float(np.mean(np.square(s))))),
        final_price_mae=("final_abs_price_error", "mean"),
        mean_predicted_return_pct=("predicted_final_return_pct", "mean"),
        mean_actual_return_pct=("actual_final_return_pct", "mean"),
    )
    out = final_metrics.join(path_metrics).reset_index()
    return out[
        [
            "step",
            "slot_label",
            "as_of_ts",
            "rows",
            "direction_accuracy",
            "final_mse_loss",
            "final_horizon_rmse",
            "final_horizon_mae",
            "path_mse_loss",
            "path_rmse",
            "path_mae",
            "final_price_rmse",
            "final_price_mae",
            "price_path_rmse",
            "price_path_mae",
            "mean_predicted_return_pct",
            "mean_actual_return_pct",
        ]
    ]


def build_model_comparison(step_metrics: pd.DataFrame) -> pd.DataFrame:
    picks = step_metrics[step_metrics["step"].isin([0, HORIZON_COUNT - 1])].copy()
    labels = {
        0: "base_view_step_00",
        HORIZON_COUNT - 1: "warm_refresh_step_25",
    }
    picks.insert(0, "model_view", picks["step"].map(labels))
    return picks


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-style Markdown table without tabulate."""
    if df.empty:
        return "_No rows._"

    text = df.astype(str)
    headers = list(text.columns)
    rows = text.values.tolist()

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown_report(
    output_dir: Path,
    forecast_date: str,
    truth_date: str,
    artifacts: pd.DataFrame,
    comparison: pd.DataFrame,
    step_metrics: pd.DataFrame,
    top_predictions: pd.DataFrame,
    artifact_notes: list[str],
) -> Path:
    report_path = output_dir / "submission_tables.md"
    comparison_md = comparison.copy()
    comparison_md["direction_accuracy"] = comparison_md["direction_accuracy"].map(format_percent)
    for col in [
        "final_mse_loss",
        "final_horizon_rmse",
        "final_horizon_mae",
        "path_mse_loss",
        "path_rmse",
        "path_mae",
        "final_price_rmse",
        "final_price_mae",
        "mean_predicted_return_pct",
        "mean_actual_return_pct",
    ]:
        comparison_md[col] = comparison_md[col].map(lambda x: f"{float(x):.6f}")

    step_md = step_metrics[["step", "slot_label", "rows", "direction_accuracy", "final_horizon_rmse", "final_mse_loss"]].copy()
    step_md["direction_accuracy"] = step_md["direction_accuracy"].map(format_percent)
    step_md["final_horizon_rmse"] = step_md["final_horizon_rmse"].map(lambda x: f"{float(x):.6f}")
    step_md["final_mse_loss"] = step_md["final_mse_loss"].map(lambda x: f"{float(x):.6f}")

    sample_md = top_predictions.copy()
    for col in ["predicted_final_return_pct", "actual_final_return_pct", "final_abs_log_error"]:
        sample_md[col] = sample_md[col].map(lambda x: f"{float(x):.6f}")
    sample_md["correct_direction"] = sample_md["correct_direction"].map(lambda x: "yes" if bool(x) else "no")

    lines = [
        "# Model Prediction Accuracy and Loss Tables",
        "",
        f"- Forecast origin date: `{forecast_date}`",
        f"- Truth date: `{truth_date}`",
        "- Return/loss unit: log return. Percent-return columns are provided for readability.",
        "- `base_view_step_00` is the app's base simulation view; `warm_refresh_step_25` is the final intraday refresh view.",
    ]
    lines.extend(f"- {note}" for note in artifact_notes)
    lines.extend(
        [
            "",
            "## Model Artifacts",
            "",
            markdown_table(artifacts),
            "",
            "## Base vs Warm Refresh",
            "",
            markdown_table(comparison_md[
                [
                    "model_view",
                    "step",
                    "slot_label",
                    "rows",
                    "direction_accuracy",
                    "final_mse_loss",
                    "final_horizon_rmse",
                    "final_horizon_mae",
                    "path_mse_loss",
                    "path_rmse",
                    "path_mae",
                    "final_price_rmse",
                    "final_price_mae",
                    "mean_predicted_return_pct",
                    "mean_actual_return_pct",
                ]
            ]),
            "",
            "## Accuracy and Loss by Refresh Step",
            "",
            markdown_table(step_md),
            "",
            "## Top Warm-Refresh Predictions With Actual Outcomes",
            "",
            markdown_table(sample_md),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def collect_artifact_metadata(repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_view, rel in [
        ("base_artifact", "model_artifacts/nibi_2026-04-16_job12292965/base/metadata.json"),
        ("warm_current_artifact", "model_artifacts/nibi_2026-04-16_job12292965/current/metadata.json"),
    ]:
        path = repo_root / rel
        if not path.exists():
            continue
        metadata = read_json(path)
        metrics = metadata.get("metrics", {})
        metric_values = [metrics.get(key) for key in ["mean_rmse", "mean_mae", "direction_accuracy"]]
        rows.append(
            {
                "model_view": model_view,
                "artifact_path": rel.removesuffix("/metadata.json"),
                "model_id": metadata.get("model_id"),
                "train_profile": metadata.get("train_profile"),
                "production_mode": metadata.get("production_mode"),
                "training_date": metadata.get("training_date"),
                "effective_as_of_date": metadata.get("effective_as_of_date"),
                "n_rows": metadata.get("n_rows"),
                "n_features": metadata.get("n_features"),
                "training_runtime_sec": round(float(metadata.get("training_runtime_sec", float("nan"))), 3),
                "stored_cv_metrics_available": not all(pd.isna(value) for value in metric_values),
            }
        )
    return pd.DataFrame(rows)


def artifact_notes(repo_root: Path) -> list[str]:
    notes: list[str] = []
    for label, rel in [
        ("base artifact", "model_artifacts/nibi_2026-04-16_job12292965/base/metadata.json"),
        ("warm/current artifact", "model_artifacts/nibi_2026-04-16_job12292965/current/metadata.json"),
    ]:
        path = repo_root / rel
        if not path.exists():
            continue
        metrics = read_json(path).get("metrics", {})
        blank = all(pd.isna(metrics.get(key)) for key in ["mean_rmse", "mean_mae", "direction_accuracy"])
        if blank:
            notes.append(f"{label} training metadata has blank/NaN CV metrics, so this report computes evaluation metrics from simulation predictions and actual bars.")
    return notes


def main() -> None:
    inputs = parse_args()
    simulation_dir = inputs.simulation_dir.resolve()
    output_dir = inputs.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_date, truth_date = infer_dates(simulation_dir, inputs.forecast_date, inputs.truth_date)
    predictions = load_predictions(simulation_dir)
    bars = load_actual_bars(inputs, forecast_date, truth_date)
    scored, long_df = build_scored_tables(predictions, bars, forecast_date, truth_date)
    step_metrics = aggregate_step_metrics(scored, long_df)
    comparison = build_model_comparison(step_metrics)
    artifact_metadata = collect_artifact_metadata(REPO_ROOT)

    top_predictions = (
        scored[scored["step"] == HORIZON_COUNT - 1]
        .sort_values("predicted_final_log_return", ascending=False)
        .head(20)
        [
            [
                "symbol",
                "slot_label",
                "cutoff_close",
                "predicted_final_close",
                "actual_final_close",
                "predicted_final_return_pct",
                "actual_final_return_pct",
                "predicted_direction",
                "actual_direction",
                "correct_direction",
                "final_abs_log_error",
            ]
        ]
        .reset_index(drop=True)
    )

    scored_out = scored[
        [
            "symbol",
            "step",
            "slot_label",
            "as_of_ts",
            "cutoff_close",
            "predicted_final_close",
            "actual_final_close",
            "predicted_final_return_pct",
            "actual_final_return_pct",
            "predicted_direction",
            "actual_direction",
            "correct_direction",
            "final_log_error",
            "final_abs_log_error",
            "final_squared_log_error",
        ]
    ].sort_values(["step", "predicted_final_return_pct"], ascending=[True, False])

    step_metrics.to_csv(output_dir / "step_metrics.csv", index=False)
    comparison.to_csv(output_dir / "base_vs_warm_refresh_metrics.csv", index=False)
    scored_out.to_csv(output_dir / "symbol_predictions_with_actuals.csv", index=False)
    top_predictions.to_csv(output_dir / "top_warm_refresh_predictions.csv", index=False)
    artifact_metadata.to_csv(output_dir / "model_artifact_metadata.csv", index=False)

    report_path = write_markdown_report(
        output_dir=output_dir,
        forecast_date=forecast_date,
        truth_date=truth_date,
        artifacts=artifact_metadata,
        comparison=comparison,
        step_metrics=step_metrics,
        top_predictions=top_predictions,
        artifact_notes=artifact_notes(REPO_ROOT),
    )
    print(f"Wrote {report_path}")
    print(f"Wrote {output_dir / 'base_vs_warm_refresh_metrics.csv'}")
    print(f"Wrote {output_dir / 'step_metrics.csv'}")
    print(f"Wrote {output_dir / 'symbol_predictions_with_actuals.csv'}")


if __name__ == "__main__":
    main()
