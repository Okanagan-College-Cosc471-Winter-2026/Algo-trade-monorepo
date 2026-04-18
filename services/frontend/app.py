from __future__ import annotations

import math
import os
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api import (
    ApiError,
    build_snapshot,
    download_snapshot,
    get_ohlc,
    health_check,
    list_snapshots,
    list_stocks,
    ops_data_freshness,
    ops_nibi_exec,
    ops_nibi_relogin,
    ops_nibi_ssh,
    ops_pipeline_logs,
    ops_status,
    predict,
    sim_base,
    sim_history,
    sim_ohlc,
    sim_session,
    sim_step,
    sim_symbols,
)

st.set_page_config(
    page_title="MarketSight",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Hide Streamlit's top header bar entirely */
    [data-testid="stHeader"] { display: none !important; }
    /* Reclaim the space that header occupied */
    .block-container {padding-top: 0.75rem; padding-bottom: 0.5rem;}
    /* Sidebar branding block */
    [data-testid="stSidebar"] .sidebar-brand {
        padding: 0.4rem 0 0.9rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.2);
        margin-bottom: 0.6rem;
    }
    /* Make plotly charts not clip at bottom */
    [data-testid="stPlotlyChart"] { overflow: visible !important; }
    /* Reduce gap between elements */
    [data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
    /* Slider: no bottom margin */
    [data-testid="stSlider"] { margin-bottom: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_H = 580   # chart height in px
TABLE_H = 400
NIBI_SIM_DIR = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")


# ── Caches ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_stocks() -> list[dict]:
    return list_stocks()

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlc(symbol: str, days: int) -> list[dict]:
    return get_ohlc(symbol, days)

@st.cache_data(ttl=60, show_spinner=False)
def load_snapshots() -> dict:
    return list_snapshots()

@st.cache_data(ttl=3600, show_spinner=False)
def load_sim_symbols() -> list[str]:
    return sim_symbols()

@st.cache_data(ttl=3600, show_spinner=False)
def load_sim_session() -> dict:
    return sim_session()

@st.cache_data(ttl=3600, show_spinner=False)
def load_sim_base(symbol: str) -> dict:
    return sim_base(symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def load_sim_step(symbol: str, step: int) -> dict:
    return sim_step(symbol, step)

@st.cache_data(ttl=3600, show_spinner=False)
def load_sim_history(symbol: str) -> list[dict]:
    return sim_history(symbol)

@st.cache_data(ttl=60, show_spinner=False)
def load_sim_ohlc(symbol: str) -> list[dict]:
    return sim_ohlc(symbol)


# ── Data helpers ──────────────────────────────────────────────────────────────

def stocks_df(stocks: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(stocks)
    if df.empty:
        return df
    for col in ("sector", "industry", "exchange"):
        if col in df.columns:
            df[col] = df[col].fillna("N/A")
    return df.sort_values("symbol").reset_index(drop=True)

def ohlc_df(symbol: str, days: int) -> pd.DataFrame:
    raw = load_ohlc(symbol, days)
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["axis_label"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
    return df.sort_values("date").reset_index(drop=True)


# ── Chart builders ────────────────────────────────────────────────────────────

def build_price_chart(df: pd.DataFrame, title: str, prediction: dict | None = None) -> go.Figure:
    """Candlestick + volume chart. Prediction path overlaid as dotted amber line."""
    fig = go.Figure()

    # Separate history from prediction date
    pred_date_str = (prediction or {}).get("prediction_date", "")[:10]
    hist = df[df["date"].dt.date.astype(str) < pred_date_str] if pred_date_str else df

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist["axis_label"],
        open=hist["open"], high=hist["high"],
        low=hist["low"],   close=hist["close"],
        name="OHLC",
        increasing_line_color="#0f766e", decreasing_line_color="#b91c1c",
        increasing_fillcolor="#14b8a6",  decreasing_fillcolor="#ef4444",
    ))

    # Volume bars on secondary Y axis
    fig.add_trace(go.Bar(
        x=hist["axis_label"], y=hist["volume"],
        name="Volume", marker_color="#94a3b8",
        opacity=0.18, yaxis="y2",
    ))

    # Prediction path
    if prediction:
        path = prediction.get("path", [])
        if path and pred_date_str:
            path_x = [f"{pred_date_str} {b['bar_time']}" for b in path]
            path_y = [b["pred_close"] for b in path]
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode="lines+markers", name="Predicted Path",
                line={"color": "#f59e0b", "width": 2, "dash": "dot"},
                marker={"size": 4, "color": "#f59e0b"},
            ))
            end_price = path_y[-1]
            fig.add_annotation(
                x=path_x[-1], y=end_price,
                text=f"Pred EOD ${end_price:.2f}",
                showarrow=True, arrowhead=2, ax=32, ay=-40,
                bgcolor="rgba(245,158,11,0.12)", bordercolor="#f59e0b",
                font={"size": 11, "color": "#92400e"},
            )

    latest_close = float(hist["close"].iloc[-1]) if not hist.empty else 0
    fig.update_layout(
        title=title, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="#fcfcfd",
        xaxis_title=None, yaxis_title="Price",
        yaxis2={"title": "Volume", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        height=CHART_H, hovermode="x unified", dragmode="pan",
    )
    fig.update_xaxes(
        showgrid=False,
        rangeslider_visible=False,
        type="category",
        nticks=20,
        tickangle=-45,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    if latest_close:
        fig.add_hline(y=latest_close, line_width=1, line_dash="dot", line_color="#94a3b8")
    return fig


def build_sim_chart(
    hist_df: pd.DataFrame,
    pred_active: dict,
    anchor_close: float | None,
    is_warm: bool,
    current_step: int,
    step_label: str | None,
    base_trees: int,
    total_trees: int,
    replay_date: str = "2026-04-07",
    ohlc_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Simulation chart: 5-day history candlesticks + volume + prediction path."""
    pre_sim = hist_df[hist_df["trade_date"] < replay_date]
    on_sim  = hist_df[hist_df["trade_date"] == replay_date]
    live_ohlc = ohlc_df if (ohlc_df is not None and not ohlc_df.empty) else None
    sim_df    = live_ohlc if live_ohlc is not None else on_sim

    # category x — equidistant per bar, inherently skips holidays/weekends/overnight
    sim_axis  = sim_df["axis_label"].reset_index(drop=True)

    # Tick at first bar of each trading day → show only the date label
    all_df = pd.concat([pre_sim, sim_df], ignore_index=True).sort_values("date")
    day_groups = all_df.groupby(all_df["date"].dt.date)["axis_label"].first()
    tickvals = day_groups.values.tolist()
    ticktext = [pd.Timestamp(str(d)).strftime("%b %d") for d in day_groups.index]

    fig = go.Figure()

    # Historical candlesticks (gray palette)
    if not pre_sim.empty:
        fig.add_trace(go.Candlestick(
            x=pre_sim["axis_label"],
            open=pre_sim["open"], high=pre_sim["high"],
            low=pre_sim["low"], close=pre_sim["close"],
            name="Historical",
            increasing_line_color="#475569", decreasing_line_color="#475569",
            increasing_fillcolor="#64748b",  decreasing_fillcolor="#64748b",
        ))
        fig.add_trace(go.Bar(
            x=pre_sim["axis_label"], y=pre_sim["volume"],
            name="Vol (hist)", marker_color="#94a3b8",
            opacity=0.18, yaxis="y2",
        ))

    # Sim-day candlesticks
    if not sim_df.empty:
        if is_warm:
            obs  = sim_df.iloc[:current_step + 1]
            rest = sim_df.iloc[current_step + 1:]
            if not obs.empty:
                fig.add_trace(go.Candlestick(
                    x=obs["axis_label"],
                    open=obs["open"], high=obs["high"],
                    low=obs["low"], close=obs["close"],
                    name=f"Apr 7 observed (→ {step_label})",
                    increasing_line_color="#0f766e", decreasing_line_color="#b91c1c",
                    increasing_fillcolor="#14b8a6",  decreasing_fillcolor="#ef4444",
                ))
            if not rest.empty:
                fig.add_trace(go.Candlestick(
                    x=rest["axis_label"],
                    open=rest["open"], high=rest["high"],
                    low=rest["low"], close=rest["close"],
                    name="Apr 7 actual (not yet seen)",
                    increasing_line_color="#7dd3fc", decreasing_line_color="#7dd3fc",
                    increasing_fillcolor="#bae6fd",  decreasing_fillcolor="#bae6fd",
                    opacity=0.55,
                ))
        else:
            fig.add_trace(go.Candlestick(
                x=sim_df["axis_label"],
                open=sim_df["open"], high=sim_df["high"],
                low=sim_df["low"], close=sim_df["close"],
                name="Apr 7 actual",
                increasing_line_color="#0f766e", decreasing_line_color="#b91c1c",
                increasing_fillcolor="#14b8a6",  decreasing_fillcolor="#ef4444",
            ))
        fig.add_trace(go.Bar(
            x=sim_df["axis_label"], y=sim_df["volume"],
            name="Vol (Apr 7)", marker_color="#0ea5e9",
            opacity=0.25, yaxis="y2",
        ))

    # Prediction path (amber dotted)
    bars = pred_active.get("bars", [])
    if bars and not sim_df.empty:
        if is_warm:
            actual_at_step = float(sim_df["close"].iloc[current_step]) if current_step < len(sim_df) else None
            if actual_at_step is not None:
                base_log = bars[current_step]["pred_log_return"]
                fwd_bars = bars[current_step:]
                fwd_xs = [sim_axis.iloc[current_step + i]
                          for i in range(len(fwd_bars))
                          if current_step + i < len(sim_axis)]
                fwd_ys = [
                    round(actual_at_step * math.exp(b["pred_log_return"] - base_log), 4)
                    for b in fwd_bars[:len(fwd_xs)]
                ]
                fig.add_trace(go.Scatter(
                    x=fwd_xs, y=fwd_ys,
                    mode="lines+markers",
                    name=f"Warm Prediction @ {step_label} ({total_trees:,} trees)",
                    line={"color": "#f59e0b", "width": 2.5, "dash": "dot"},
                    marker={"size": 4, "color": "#f59e0b"},
                ))
        elif anchor_close:
            pred_xs = [sim_axis.iloc[i] for i in range(len(bars)) if i < len(sim_axis)]
            pred_ys = [round(anchor_close * math.exp(b["pred_log_return"]), 4)
                       for b in bars[:len(pred_xs)]]
            fig.add_trace(go.Scatter(
                x=pred_xs, y=pred_ys,
                mode="lines+markers",
                name=f"Base Prediction ({base_trees:,} trees)",
                line={"color": "#f59e0b", "width": 2.5, "dash": "dot"},
                marker={"size": 4, "color": "#f59e0b"},
            ))

    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="#fcfcfd",
        xaxis_title=None, yaxis_title="Price (USD)",
        yaxis2={
            "overlaying": "y", "side": "right",
            "showgrid": False, "showticklabels": False,
            "fixedrange": True,
        },
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=CHART_H, hovermode="x unified", dragmode="pan",
    )
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        rangeslider_visible=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig




# ── Page fragments (only these rerun on widget change inside them) ─────────────

@st.fragment
def stocks_chart_fragment(symbol: str, days: int, detail: dict) -> None:
    """Chart + prediction toggle — reruns in isolation; no page scroll."""
    show_pred = st.toggle("Overlay model prediction", value=False, key=f"pred_toggle_{symbol}")
    prediction = None
    if show_pred:
        with st.spinner("Running inference..."):
            try:
                prediction = predict(symbol)
            except ApiError as exc:
                st.error(str(exc))

    if prediction:
        path = prediction.get("path", [])
        end_price = path[-1]["pred_close"] if path else prediction["current_price"]
        full_ret = prediction.get("predicted_full_day_return", 0.0)
        direction = prediction.get("predicted_direction", "—")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Price", f"${prediction['current_price']:.2f}")
        c2.metric("Predicted EOD", f"${end_price:.2f}")
        c3.metric("Full-Day Return", f"{full_ret:.2f}%",
                  delta=f"{'▲' if direction=='up' else '▼'} {direction}")
        c4.metric("Model", prediction["model_version"])

    df = ohlc_df(symbol, days)
    if df.empty:
        st.warning("No OHLC data available for this symbol.")
        return
    fig = build_price_chart(df, f"{detail.get('name', symbol)} ({symbol})", prediction)
    st.plotly_chart(fig, use_container_width=True)

    info_tab, data_tab = st.tabs(["Summary", "Raw Data"])
    with info_tab:
        c1, c2 = st.columns(2)
        c1.write(f"**Sector:** {detail.get('sector') or 'N/A'}")
        c1.write(f"**Industry:** {detail.get('industry') or 'N/A'}")
        c1.write(f"**Exchange:** {detail.get('exchange') or 'N/A'}")
        c2.write(f"**High:** ${float(df['high'].max()):.2f}")
        c2.write(f"**Low:** ${float(df['low'].min()):.2f}")
        c2.write(f"**Latest Vol:** {int(df['volume'].iloc[-1]):,}")
    with data_tab:
        raw = df[["date","open","high","low","close","volume"]].tail(200).sort_values("date", ascending=False)
        st.dataframe(raw, use_container_width=True, hide_index=True, height=360,
            column_config={
                "date": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "open": st.column_config.NumberColumn("Open", format="$%.2f"),
                "high": st.column_config.NumberColumn("High", format="$%.2f"),
                "low": st.column_config.NumberColumn("Low", format="$%.2f"),
                "close": st.column_config.NumberColumn("Close", format="$%.2f"),
                "volume": st.column_config.NumberColumn("Volume", format="%d"),
            })


@st.fragment
def sim_fragment(
    symbol: str,
    hist_df: pd.DataFrame,
    pred_base: dict,
    session_info: dict,
    anchor_close: float | None,
    actual_ret: float | None,
    ohlc_df: pd.DataFrame | None = None,
) -> None:
    """Slider + chart in one fragment — dragging never scrolls the page."""
    step_labels: list[str] = session_info.get("step_labels", [])
    step_count: int = session_info.get("steps_completed", 26)
    base_trees: int = session_info.get("base_trees", 1157)
    warm_per_step: int = session_info.get("warm_trees_per_step", 30)

    is_warm = st.session_state.get("sim_mode") == "Warm-Refresh Simulation"
    # note: "Base Model (Apr 6 → Apr 7)" is the non-warm option; any non-warm value lands here

    pred_active = pred_base
    current_step = 0
    step_label: str | None = None
    total_trees = base_trees

    if is_warm:
        current_step = st.slider(
            "Intraday bar (drag to step through warm-refresh)",
            min_value=0, max_value=step_count - 1, value=0,
            key="sim_step_slider",
            help="Each step adds warm-refresh trees trained on bars observed so far.",
        )
        step_label = step_labels[current_step] if current_step < len(step_labels) else str(current_step)
        total_trees = base_trees + (current_step + 1) * warm_per_step
        with st.spinner(f"Loading step {step_label}…"):
            try:
                pred_active = load_sim_step(symbol, current_step)
            except ApiError as exc:
                st.error(str(exc))
                return

    replay_date = session_info.get("replay_date", "—")
    eff_date = session_info.get("effective_as_of_date", "—")

    # Metrics
    full_ret = pred_active.get("predicted_full_day_return", 0.0)
    direction = pred_active.get("predicted_direction", "—")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Trained", eff_date)
    c2.metric("Target Date", replay_date)
    c3.metric("Predicted Return", f"{full_ret:+.4f}%")
    c4.metric("Direction", direction.upper())
    if actual_ret is not None:
        label = f"step {current_step} ({step_label}), {total_trees:,} trees" if is_warm else f"base, {base_trees:,} trees"
        st.caption(f"Actual {replay_date}: **{actual_ret:+.2f}%** | Model ({label}): **{full_ret:+.4f}%**")

    # Chart
    fig = build_sim_chart(
        hist_df, pred_active, anchor_close, is_warm,
        current_step, step_label, base_trees, total_trees,
        replay_date=replay_date,
        ohlc_df=ohlc_df,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Pages ──────────────────────────────────────────────────────────────────────

def render_overview(stocks: list[dict]) -> None:
    st.subheader("Overview")
    try:
        healthy = health_check()
        health_error = None
    except Exception as exc:  # noqa: BLE001
        healthy = False
        health_error = str(exc)

    df = stocks_df(stocks)
    sectors = sorted({s["sector"] for s in stocks if s.get("sector") and s["sector"] != "N/A"})

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked Stocks", len(stocks))
    c2.metric("Sectors", len(sectors))
    c3.metric("API Status", "Online" if healthy else "Unavailable")
    if health_error:
        st.warning(f"Health check failed: {health_error}")

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Coverage")
        available_cols = [c for c in ["symbol","name","sector","exchange"] if c in df.columns]
        st.dataframe(
            df[available_cols] if available_cols else df,
            use_container_width=True, hide_index=True, height=460,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "name": st.column_config.TextColumn("Company", width="medium"),
                "sector": st.column_config.TextColumn("Sector", width="medium"),
                "exchange": st.column_config.TextColumn("Exchange", width="small"),
            },
        )
    with right:
        st.markdown("#### Sector Breakdown")
        if sectors:
            sector_counts = df[df["sector"] != "N/A"]["sector"].value_counts()
            st.bar_chart(sector_counts)
        else:
            st.info("No sector metadata available.")


def render_stocks(stocks: list[dict], symbol: str, days: int) -> None:
    if not stocks:
        st.info("No stocks available.")
        return
    detail = next((s for s in stocks if s["symbol"] == symbol), {})

    # Top metrics (outside fragment — only reruns when symbol/days change)
    df = ohlc_df(symbol, days)
    if not df.empty:
        latest = float(df["close"].iloc[-1])
        first  = float(df["close"].iloc[0])
        period_ret = ((latest / first) - 1) * 100 if first else 0.0
        avg_vol = float(df["volume"].tail(20).mean())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Close", f"${latest:.2f}")
        c2.metric("Period Return", f"{period_ret:.2f}%")
        c3.metric("Avg Vol (20)", f"{int(avg_vol):,}")
        c4.metric("Bars", f"{len(df):,}")

    # Fragment handles prediction toggle + chart (no scroll on toggle)
    stocks_chart_fragment(symbol, days, detail)


def render_predictions(stocks: list[dict], symbol: str) -> None:
    if not stocks:
        st.info("No stocks available.")
        return
    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Running inference…"):
            try:
                payload = predict(symbol)
            except ApiError as exc:
                st.error(str(exc))
                return
        path = payload.get("path", [])
        end_price = path[-1]["pred_close"] if path else payload["current_price"]
        full_ret = payload.get("predicted_full_day_return", 0.0)
        direction = payload.get("predicted_direction", "—")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Price", f"${payload['current_price']:.2f}")
        c2.metric("Predicted EOD", f"${end_price:.2f}")
        c3.metric("Return", f"{full_ret:.2f}%",
                  delta=f"{'▲' if direction=='up' else '▼'} {direction}")
        c4.metric("Model", payload["model_version"])

        df = ohlc_df(symbol, 365)
        if not df.empty:
            st.markdown(f"#### {symbol} — Predicted Path")
            fig = build_price_chart(df, f"{symbol} — Predicted Path", payload)
            st.plotly_chart(fig, use_container_width=True)
        st.caption(f"26-bar 15-min path for {payload['prediction_date'][:10]}.")


def render_simulation(stocks: list[dict], symbol: str) -> None:
    if not symbol:
        st.markdown(
            """
            <div style="
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; padding:4rem 2rem; text-align:center;
                color:#64748b;
            ">
                <div style="font-size:2.5rem; margin-bottom:0.75rem;">📂</div>
                <div style="font-size:1.1rem; font-weight:600; margin-bottom:0.4rem; color:#334155;">
                    Simulation data not available
                </div>
                <div style="font-size:0.9rem; max-width:480px; line-height:1.6;">
                    Prediction CSVs were generated on the remote HPC cluster (NIBI/Compute Canada)
                    and are not committed to this repository.
                    Run <code>promote_model.sh</code> or sync the
                    <code>model_artifacts/simulation_*</code> directory to enable this view.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        session_info = load_sim_session()
    except ApiError as exc:
        st.error(f"Could not load session info: {exc}")
        return

    replay_date = session_info.get("replay_date", "")
    eff_date = session_info.get("effective_as_of_date", "")

    try:
        hist_raw = load_sim_history(symbol)
        hist = pd.DataFrame(hist_raw)
        hist["date"] = pd.to_datetime(hist["time"], unit="s", utc=True)
        hist["axis_label"] = hist["date"].dt.strftime("%Y-%m-%d %H:%M")
        hist = hist.sort_values("date").reset_index(drop=True)
    except ApiError as exc:
        st.error(f"Could not load history: {exc}")
        return

    # anchor = last bar of the training cutoff day (effective_as_of_date from base metadata)
    anchor_rows = hist[hist["trade_date"] == eff_date] if eff_date else pd.DataFrame()
    anchor_close = float(anchor_rows["close"].iloc[-1]) if not anchor_rows.empty else None

    # actual return on the simulation day (open→close)
    sim_day = hist[hist["trade_date"] == replay_date] if replay_date else pd.DataFrame()
    actual_ret = None
    if not sim_day.empty:
        actual_ret = (float(sim_day["close"].iloc[-1]) / float(sim_day["close"].iloc[0]) - 1) * 100

    # Live OHLC from DB for the simulation day (real candlestick bars)
    try:
        ohlc_raw = load_sim_ohlc(symbol)
        ohlc = pd.DataFrame(ohlc_raw)
        if not ohlc.empty:
            ohlc["date"] = pd.to_datetime(ohlc["time"], unit="s", utc=True)
            ohlc["axis_label"] = ohlc["date"].dt.strftime("%Y-%m-%d %H:%M")
            ohlc = ohlc.sort_values("date").reset_index(drop=True)
    except ApiError:
        ohlc = pd.DataFrame()

    try:
        pred_base = load_sim_base(symbol)
    except ApiError as exc:
        st.error(f"Base prediction failed: {exc}")
        return

    # Fragment owns the slider + chart — no full-page reruns when dragging
    sim_fragment(symbol, hist, pred_base, session_info, anchor_close, actual_ret, ohlc)


def render_snapshots() -> None:
    with st.form("build_snapshot"):
        ticker = st.text_input("Ticker", value="ALL", help="Use ALL for every stock.")
        left, right = st.columns(2)
        start_date = left.text_input("Start date", placeholder="YYYY-MM-DD")
        end_date   = right.text_input("End date",   placeholder="YYYY-MM-DD")
        file_format = st.selectbox("Format", ["parquet", "csv", "both"])
        submitted = st.form_submit_button("Build Snapshot", type="primary")

    if submitted:
        with st.spinner("Building…"):
            try:
                result = build_snapshot({
                    "ticker": ticker or "ALL",
                    "start_date": start_date or None,
                    "end_date": end_date or None,
                    "format": file_format,
                })
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.success(f"Created for {result['tickers_processed']} ticker(s), {result['total_rows_extracted']} rows.")
                st.json(result, expanded=False)
                load_snapshots.clear()

    try:
        payload = load_snapshots()
    except ApiError as exc:
        st.error(str(exc))
        return

    snapshots = payload.get("snapshots", [])
    st.caption(f"Directory: {payload.get('directory', '—')}")
    if not snapshots:
        st.info("No snapshots yet.")
        return

    df = pd.DataFrame(snapshots).sort_values("filename").reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True, height=TABLE_H,
        column_config={
            "filename": st.column_config.TextColumn("File", width="large"),
            "size_mb": st.column_config.NumberColumn("Size (MB)", format="%.2f"),
        })

    selected = st.selectbox("Download", df["filename"].tolist())
    if st.button("Prepare Download"):
        with st.spinner("Fetching…"):
            try:
                file_obj = download_snapshot(selected)
            except ApiError as exc:
                st.error(str(exc))
                return
        st.download_button(f"Download {selected}", file_obj.getvalue(),
                           file_name=selected, mime="application/octet-stream")


# ── Ops page ──────────────────────────────────────────────────────────────────

def _status_badge(ok: bool, label_ok: str, label_bad: str) -> str:
    col = "#16a34a" if ok else "#dc2626"
    label = label_ok if ok else label_bad
    return (
        f'<span style="background:{col};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600">{label}</span>'
    )


def render_ops() -> None:
    st.subheader("System Operations")

    # ── Fetch snapshot ─────────────────────────────────────────────
    with st.spinner("Loading ops status…"):
        try:
            snap = ops_status()
        except ApiError as exc:
            st.error(f"Could not reach backend: {exc}")
            return

    gen_at = snap.get("generated_at", "")
    st.caption(f"Snapshot at {gen_at[:19].replace('T',' ')} UTC — refresh page to update")

    # ══ 1. Service Health ══════════════════════════════════════════
    st.markdown("#### Service Health")
    c1, c2, c3, c4 = st.columns(4)

    # Backend (we got here so it's up)
    c1.markdown("**Backend API**")
    c1.markdown(_status_badge(True, "Online", "Offline"), unsafe_allow_html=True)

    # SSH socket → NIBI
    ssh = snap.get("ssh_socket", {})
    c2.markdown("**NIBI SSH Socket**")
    c2.markdown(_status_badge(ssh.get("alive", False), "Socket alive", "Socket dead"), unsafe_allow_html=True)
    if not ssh.get("alive"):
        c2.caption("Run `morning_login.sh`")

    # Collector
    col = snap.get("collector", {})
    collector_ok = col.get("last_status") == "success" and (col.get("age_min") or 9999) < 30
    c3.markdown("**Collector Pipeline**")
    c3.markdown(_status_badge(collector_ok, "Running", "Stale / Error"), unsafe_allow_html=True)
    if col.get("age_min") is not None:
        c3.caption(f"Last run: {col['age_min']:.0f} min ago — {col.get('last_stage','?')} [{col.get('last_status','?')}]")
    elif col.get("error"):
        c3.caption(f"DB error: {col['error']}")

    # Data freshness
    data = snap.get("data", {})
    stale = data.get("staleness_min")
    data_ok = stale is not None and stale < 20
    c4.markdown("**Market Data**")
    c4.markdown(_status_badge(data_ok, "Fresh", "Stale"), unsafe_allow_html=True)
    if stale is not None:
        c4.caption(f"Last bar: {stale:.0f} min ago  |  {data.get('total_rows',0):,} rows")

    st.divider()

    # ══ 2. NIBI Job Status ═════════════════════════════════════════
    st.markdown("#### NIBI Training Job")
    job = snap.get("nibi_job", {})
    model = snap.get("model", {})
    live_state = job.get("live_state") or job.get("status", "unknown")

    state_colors = {
        "RUNNING": "#16a34a", "COMPLETED": "#0284c7", "completed": "#0284c7",
        "PENDING": "#d97706", "submitted": "#d97706",
        "FAILED": "#dc2626", "CANCELLED": "#dc2626", "TIMEOUT": "#dc2626",
    }
    state_col = state_colors.get(live_state, "#64748b")

    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Job ID", job.get("job_id") or "—")
    j2.metric("Sim Date", job.get("sim_date") or "—")
    j3.markdown(f"**Status**")
    j3.markdown(
        f'<span style="background:{state_col};color:white;padding:3px 12px;'
        f'border-radius:12px;font-weight:700">{live_state.upper()}</span>',
        unsafe_allow_html=True,
    )
    submitted_at = job.get("submitted_at", "")
    j4.metric("Submitted", submitted_at[11:19] + " UTC" if len(submitted_at) > 18 else submitted_at or "—")

    # Window progress bar
    w_ok    = model.get("windows_ok", 0)
    w_total = model.get("windows_total", 26)
    w_err   = model.get("windows_error", 0)
    prog_val = w_ok / w_total if w_total else 0
    st.markdown(f"**Warm-refresh windows:** {w_ok} / {w_total} completed"
                + (f"  ({w_err} errors)" if w_err else ""))
    st.progress(prog_val)

    # Per-window detail (collapsible)
    steps = model.get("windows_steps", [])
    if steps:
        with st.expander("Window breakdown", expanded=False):
            rows = []
            for s in steps:
                rows.append({
                    "Step": s.get("step"),
                    "Time (ET)": s.get("et_label", ""),
                    "Status": s.get("status", ""),
                    "Train (s)": s.get("train_sec"),
                    "Total (s)": s.get("total_sec"),
                })
            df_steps = pd.DataFrame(rows)
            st.dataframe(df_steps, use_container_width=True, hide_index=True,
                column_config={
                    "Step":      st.column_config.NumberColumn(width="small"),
                    "Time (ET)": st.column_config.TextColumn(width="small"),
                    "Status":    st.column_config.TextColumn(width="medium"),
                    "Train (s)": st.column_config.NumberColumn(format="%.1f", width="small"),
                    "Total (s)": st.column_config.NumberColumn(format="%.1f", width="small"),
                })

    st.divider()

    # ══ 3. Active Model ═══════════════════════════════════════════
    st.markdown("#### Active Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bundle", model.get("name") or "—")
    m2.metric("Train Cutoff", model.get("train_end_date") or "—")
    m3.metric("Base Trees", f"{model.get('n_estimators', '?'):,}" if model.get("n_estimators") else "—")
    promoted = model.get("promoted_at", "")
    m4.metric("Promoted", promoted[11:16] + " UTC" if len(promoted) > 15 else promoted or "—")
    if not model.get("active"):
        st.warning("No active model symlink found at model_artifacts/current_base")

    st.divider()

    # ══ 4. VPS Machine Health ═════════════════════════════════════
    st.markdown("#### VPS Machine Health")
    machine = snap.get("machine", {})

    # ── Top summary row ───────────────────────────────────────────
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Host", machine.get("hostname", "—"))
    h2.metric("Uptime", f"{machine.get('uptime_hrs', 0):.0f} hrs")
    load = machine.get("load_avg", {})
    h3.metric("Load avg", f"{load.get('1m', 0):.2f} / {load.get('5m', 0):.2f} / {load.get('15m', 0):.2f}",
              help="1m / 5m / 15m load average")
    h4.metric("Processes", machine.get("process_count", "—"))
    h5.metric("OS", machine.get("os", "—"))

    st.markdown("")

    # ── CPU ───────────────────────────────────────────────────────
    cpu_pct = machine.get("cpu_pct", 0)
    cpu_cores = machine.get("cpu_cores", 1)
    cpu_col = "#16a34a" if cpu_pct < 70 else "#d97706" if cpu_pct < 90 else "#dc2626"
    st.markdown(f"**CPU** — {cpu_pct:.1f}%  <span style='color:#64748b;font-size:0.8rem'>"
                f"{machine.get('cpu_model','')} · {cpu_cores} logical cores</span>",
                unsafe_allow_html=True)
    st.progress(cpu_pct / 100)

    # Per-core breakdown (collapsed)
    per_core = machine.get("cpu_per_core", [])
    if per_core:
        with st.expander(f"Per-core breakdown ({len(per_core)} cores)", expanded=False):
            cols = st.columns(min(len(per_core), 8))
            for i, pct in enumerate(per_core):
                c = cols[i % len(cols)]
                c.caption(f"Core {i}")
                c.progress(pct / 100)
                c.caption(f"{pct:.0f}%")

    st.markdown("")

    # ── Memory ───────────────────────────────────────────────────
    ram_pct   = machine.get("ram_pct", 0)
    ram_used  = machine.get("ram_used_gb", 0)
    ram_total = machine.get("ram_total_gb", 1)
    swap_pct  = machine.get("swap_pct", 0)
    swap_used = machine.get("swap_used_gb", 0)
    swap_tot  = machine.get("swap_total_gb", 0)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"**RAM** — {ram_pct:.0f}%  "
                    f"<span style='color:#64748b;font-size:0.8rem'>{ram_used:.1f} / {ram_total:.1f} GB</span>",
                    unsafe_allow_html=True)
        st.progress(ram_pct / 100)
    with mc2:
        st.markdown(f"**Swap** — {swap_pct:.0f}%  "
                    f"<span style='color:#64748b;font-size:0.8rem'>{swap_used:.1f} / {swap_tot:.1f} GB</span>",
                    unsafe_allow_html=True)
        st.progress(swap_pct / 100)

    st.markdown("")

    # ── Disk ─────────────────────────────────────────────────────
    disk = machine.get("disk", {})
    disk_cols = st.columns(len(disk) or 1)
    for i, (mount, d) in enumerate(disk.items()):
        dpct = d.get("pct", 0)
        with disk_cols[i]:
            warn = " ⚠️" if dpct > 85 else ""
            st.markdown(f"**Disk `{mount}`** — {dpct:.0f}%{warn}  "
                        f"<span style='color:#64748b;font-size:0.8rem'>"
                        f"{d.get('used_gb',0):.0f} / {d.get('total_gb',0):.0f} GB</span>",
                        unsafe_allow_html=True)
            st.progress(dpct / 100)

    st.markdown("")

    # ── Network ──────────────────────────────────────────────────
    n1, n2, n3 = st.columns(3)
    n1.metric("Net Sent",    f"{machine.get('net_sent_gb', 0):.2f} GB",
              help="Cumulative since boot")
    n2.metric("Net Received", f"{machine.get('net_recv_gb', 0):.2f} GB",
              help="Cumulative since boot")
    n3.metric("Packets (tx/rx)",
              f"{machine.get('net_pkts_sent',0):,} / {machine.get('net_pkts_recv',0):,}")

    # ── NIBI GPU (live, only when job RUNNING) ────────────────────
    nibi_gpu = snap.get("nibi_gpu")
    if nibi_gpu:
        st.markdown("")
        st.markdown("**NIBI GPU (live)**")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("GPU", nibi_gpu.get("name", "H100"))
        mem_used  = nibi_gpu.get("mem_used", 0)
        mem_total = nibi_gpu.get("mem_total", 1)
        g2.metric("VRAM", f"{mem_used:,} / {mem_total:,} MB")
        g3.metric("Utilisation", f"{nibi_gpu.get('util_pct', 0)}%")
        g4.metric("Temp", f"{nibi_gpu.get('temp_c', 0)} °C")
        vram_pct = mem_used / mem_total if mem_total else 0
        st.progress(vram_pct, text=f"VRAM {vram_pct*100:.0f}%")
    else:
        st.caption("No GPU on this VPS — training runs on NIBI H100  |  "
                   "NIBI GPU metrics appear here when a job is RUNNING")

    st.divider()

    # ══ 4b. NIBI Jobs ═════════════════════════════════════════════
    st.markdown("#### NIBI Jobs")
    nibi_jobs = snap.get("nibi_jobs", {})

    if not nibi_jobs.get("available"):
        st.warning("SSH socket offline — NIBI job data unavailable. Reconnect via SSH section below.")
    else:
        # Active / queued jobs
        queued = nibi_jobs.get("queued", [])
        if queued:
            st.markdown(f"**Active / Queued** ({len(queued)} job{'s' if len(queued) != 1 else ''})")
            state_colors = {
                "RUNNING": "#16a34a", "PENDING": "#d97706",
                "FAILED": "#dc2626", "COMPLETED": "#0284c7", "CANCELLED": "#64748b",
            }
            for j in queued:
                s = j.get("state", "")
                col = state_colors.get(s, "#64748b")
                badge = (f'<span style="background:{col};color:white;padding:1px 8px;'
                         f'border-radius:10px;font-size:0.75rem;font-weight:600">{s}</span>')
                st.markdown(
                    f"**{j.get('job_id')}** &nbsp; {j.get('name')} &nbsp; {badge} &nbsp; "
                    f"elapsed `{j.get('elapsed')}` / limit `{j.get('time_lim')}` &nbsp; "
                    f"start: `{j.get('start','—')}`",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No active or pending jobs in queue.")

        st.markdown("")

        # Recent job history
        history = nibi_jobs.get("history", [])
        if history:
            st.markdown("**Recent Job History** (last 7 days)")
            df_hist = pd.DataFrame(history)
            state_map = {
                "COMPLETED": "✅ COMPLETED", "FAILED": "❌ FAILED",
                "CANCELLED": "⛔ CANCELLED", "RUNNING": "🟢 RUNNING",
                "PENDING": "🟡 PENDING", "TIMEOUT": "⏱ TIMEOUT",
                "OUT_OF_MEMORY": "💥 OOM",
            }
            df_hist["state"] = df_hist["state"].map(lambda s: state_map.get(s, s))
            st.dataframe(df_hist, use_container_width=True, hide_index=True,
                column_config={
                    "job_id":  st.column_config.TextColumn("Job ID",  width="small"),
                    "name":    st.column_config.TextColumn("Name",    width="medium"),
                    "state":   st.column_config.TextColumn("State",   width="medium"),
                    "exit":    st.column_config.TextColumn("Exit",    width="small"),
                    "elapsed": st.column_config.TextColumn("Elapsed", width="small"),
                    "start":   st.column_config.TextColumn("Started", width="medium"),
                })

        # Scratch quota
        quota = nibi_jobs.get("quota_raw")
        if quota:
            with st.expander("NIBI scratch quota", expanded=False):
                st.code(quota, language=None)

    st.divider()

    # ══ 5. Pipeline Run History ═══════════════════════════════════
    st.markdown("#### Collector Pipeline History")
    try:
        logs = ops_pipeline_logs(limit=30)
    except ApiError:
        logs = []

    if logs:
        df_logs = pd.DataFrame(logs)
        df_logs["ts"] = pd.to_datetime(df_logs["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        status_map = {"success": "✅", "failed": "❌", "warning": "⚠️"}
        df_logs["status"] = df_logs["status"].map(lambda s: f"{status_map.get(s, '')} {s}")
        st.dataframe(df_logs, use_container_width=True, hide_index=True, height=280,
            column_config={
                "ts":      st.column_config.TextColumn("Time (UTC)", width="medium"),
                "stage":   st.column_config.TextColumn("Stage", width="medium"),
                "status":  st.column_config.TextColumn("Status", width="small"),
                "message": st.column_config.TextColumn("Detail", width="large"),
            })
    else:
        st.info("No pipeline logs available.")

    st.divider()

    # ══ 6. NIBI SSH Terminal ═══════════════════════════════════════
    st.markdown("#### NIBI Remote Commands")
    st.caption(
        "Runs over the existing SSH ControlMaster socket (no MFA needed while socket is alive). "
        "Read-only commands only: `squeue`, `sacct`, `tail`, `ls`, `nvidia-smi`, `quota`, etc."
    )

    preset_commands = {
        "My jobs in queue":           f"squeue -u {snap.get('nibi_job', {}).get('sim_date', 'harshsaw')} -o '%.10i %.9P %.20j %.8u %.8T %.10M %.6D %R' 2>/dev/null || squeue -u harshsaw",
        "Job accounting (last job)":  f"sacct -j {job.get('job_id', '0')} --format=JobID,State,Elapsed,Start,End,AllocCPUS --noheader 2>/dev/null",
        "NIBI GPU nodes":             "sinfo -p gpu --noheader -o '%n %t %C' 2>/dev/null | head -10",
        "Simulation log (tail 30)":   f"tail -30 {NIBI_SIM_DIR}/logs/sim_full_day_{job.get('job_id','0')}.out 2>/dev/null || echo 'log not found'",
        "Disk quota":                 "quota -s 2>/dev/null || df -h $HOME",
        "Custom…":                    "",
    }

    preset = st.selectbox("Quick commands", list(preset_commands.keys()), key="nibi_preset")
    default_cmd = preset_commands[preset]
    cmd_input = st.text_input("Command", value=default_cmd, key="nibi_cmd",
                              placeholder="squeue -u harshsaw")

    if st.button("Run on NIBI", type="primary", disabled=not ssh.get("alive")):
        if not ssh.get("alive"):
            st.error("SSH socket is dead — run morning_login.sh first.")
        elif cmd_input.strip():
            with st.spinner("Running…"):
                try:
                    result = ops_nibi_exec(cmd_input.strip())
                    rc = result.get("rc", -1)
                    stdout = result.get("stdout", "")
                    stderr = result.get("stderr", "")
                    if rc == 0:
                        st.success(f"Exit code: {rc}")
                    else:
                        st.warning(f"Exit code: {rc}")
                    if stdout:
                        st.code(stdout, language=None)
                    if stderr:
                        st.caption(f"stderr: {stderr}")
                except ApiError as exc:
                    st.error(str(exc))

    if not ssh.get("alive"):
        st.warning(
            "SSH socket is not active. Choose a re-auth method below:"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Option A — Auto Re-login**")
            st.caption(
                "Triggers `auto_login.py` on the backend.  \n"
                "- If `NIBI_TOTP_SECRET` is set in `.env` → **fully headless**, no phone needed.  \n"
                "- Otherwise → Duo push sent, approve on your phone, then click Refresh."
            )
            if st.button("Re-login via Backend", type="primary"):
                try:
                    resp = ops_nibi_relogin()
                    if resp.get("already_alive"):
                        st.success("Socket already active — refresh the page.")
                    else:
                        mode = resp.get("mode", "?")
                        if mode == "totp":
                            st.success("TOTP login started. Refreshing in 5s…")
                            import time; time.sleep(5)
                            st.rerun()
                        else:
                            st.info(
                                "Duo push sent. Approve on your phone, "
                                "then click **Refresh SSH Status** below."
                            )
                except ApiError as exc:
                    st.error(str(exc))

            if st.button("Refresh SSH Status"):
                try:
                    fresh = ops_nibi_ssh()
                    if fresh.get("alive"):
                        st.success("Socket is now alive! Reload the Ops page.")
                    else:
                        st.warning("Still not alive — push not approved yet?")
                except ApiError as exc:
                    st.error(str(exc))

        with col_b:
            st.markdown("**Option B — Web Terminal**")
            st.caption(
                "Open a browser terminal, run `morning_login.sh`, approve Duo push manually.  \n"
                "Use this if Option A fails (e.g. pexpect not installed in container)."
            )
            st.markdown(
                "[Open Web Terminal (port 7681)](http://localhost:7681)  \n"
                "Then run:  \n"
                "```bash\n"
                "bash ml/ml/nibi/morning_login.sh\n"
                "```"
            )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def build_sidebar(stocks: list[dict]) -> tuple[str, str, int, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div style="font-size:1.3rem;font-weight:700;color:#0f172a;line-height:1.1">
                    MarketSight
                </div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:0.25rem;line-height:1.45">
                    Market data, inference &amp; dataset snapshots.<br>
                    Optimized for fast scanning.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation", ["Overview","Stocks","Predictions","Simulation","Snapshots","Ops"],
            label_visibility="collapsed",
        )
        st.divider()

        df = stocks_df(stocks)
        symbols = df["symbol"].tolist() if not df.empty else []
        symbol = symbols[0] if symbols else ""
        days = 30



        if page in ("Stocks", "Predictions"):
            st.markdown("**Controls**")
            search = st.text_input("Search", placeholder="symbol or company", label_visibility="collapsed")
            sectors = ["All"] + sorted([s for s in df["sector"].unique() if s != "N/A"])
            sector_filter = st.selectbox("Sector", sectors)

            filtered = df.copy()
            if search:
                m = search.strip().lower()
                filtered = filtered[
                    filtered["symbol"].str.lower().str.contains(m, na=False)
                    | filtered["name"].str.lower().str.contains(m, na=False)
                ]
            if sector_filter != "All":
                filtered = filtered[filtered["sector"] == sector_filter]

            opts = filtered["symbol"].tolist() if not filtered.empty else symbols
            default = st.session_state.get("selected_symbol", opts[0] if opts else "")
            if default not in opts:
                default = opts[0] if opts else ""

            if opts:
                symbol = st.selectbox("Stock", opts,
                    index=opts.index(default) if default in opts else 0,
                    key="sb_symbol")
                st.session_state["selected_symbol"] = symbol

            if page == "Stocks":
                days = st.select_slider(
                    "History window",
                    options=[7, 30, 90, 180, 365, 730],
                    value=st.session_state.get("stocks_days", 30),
                    key="stocks_days",
                )

        elif page == "Simulation":
            st.markdown("**Simulation — 2026-04-07**")
            try:
                sim_syms = load_sim_symbols()
            except Exception:
                sim_syms = symbols  # fall back to market symbols if endpoint unavailable
            default_sym = "AAPL" if "AAPL" in sim_syms else (sim_syms[0] if sim_syms else "")
            symbol = st.selectbox("Asset", sim_syms,
                index=sim_syms.index(default_sym) if default_sym in sim_syms else 0,
                key="sim_symbol") if sim_syms else ""
            st.radio(
                "Mode",
                ["Base Model (Apr 6 → Apr 7)", "Warm-Refresh Simulation"],
                key="sim_mode",
            )
            # Step slider lives inside the fragment (main area) so dragging it
            # only reruns the fragment — no sidebar needed here.

        st.divider()
        st.caption(f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC")

    return page, symbol, days


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        stocks = load_stocks()
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to load stocks: {exc}")
        stocks = []

    page, symbol, days = build_sidebar(stocks)

    if page == "Overview":
        render_overview(stocks)
    elif page == "Stocks":
        render_stocks(stocks, symbol, days)
    elif page == "Predictions":
        render_predictions(stocks, symbol)
    elif page == "Simulation":
        render_simulation(stocks, symbol)
    elif page == "Ops":
        render_ops()
    else:
        render_snapshots()


if __name__ == "__main__":
    main()
