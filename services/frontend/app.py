from __future__ import annotations

import math
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
    /* Tighten top padding */
    .block-container {padding-top: 1rem; padding-bottom: 0.5rem;}
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
    """Simulation chart: 5-day history + Apr-7 actual + prediction path."""
    pre_sim = hist_df[hist_df["trade_date"] < replay_date]
    on_sim  = hist_df[hist_df["trade_date"] == replay_date]
    live_ohlc = ohlc_df if (ohlc_df is not None and not ohlc_df.empty) else None
    sim_df    = live_ohlc if live_ohlc is not None else on_sim
    sim_axis = sim_df["axis_label"].reset_index(drop=True)
    sim_close = sim_df["close"].reset_index(drop=True)

    fig = go.Figure()

    # Trace 1: Historical close (gray)
    if not pre_sim.empty:
        fig.add_trace(go.Scatter(
            x=pre_sim["axis_label"], y=pre_sim["close"],
            mode="lines", name="Historical Close (DB)",
            line={"color": "#64748b", "width": 2},
        ))

    # Trace 2 & 3: Apr-7 actual bars
    if not sim_df.empty:
        if is_warm:
            obs  = sim_df.iloc[:current_step + 1]
            rest = sim_df.iloc[current_step + 1:]
            fig.add_trace(go.Scatter(
                x=obs["axis_label"], y=obs["close"],
                mode="lines", name=f"Apr 7 observed (→ {step_label})",
                line={"color": "#0ea5e9", "width": 2},
            ))
            if not rest.empty:
                fig.add_trace(go.Scatter(
                    x=rest["axis_label"], y=rest["close"],
                    mode="lines", name="Apr 7 actual (not yet seen)",
                    line={"color": "#38bdf8", "width": 2, "dash": "dashdot"},
                    opacity=0.6,
                ))
        else:
            fig.add_trace(go.Scatter(
                x=sim_df["axis_label"], y=sim_df["close"],
                mode="lines", name="Apr 7 actual (DB)",
                line={"color": "#0ea5e9", "width": 2},
            ))

    # Trace 4: Prediction path (amber dotted)
    bars = pred_active.get("bars", [])
    if bars and not sim_df.empty:
        if is_warm:
            actual_at_step = float(sim_close.iloc[current_step]) if current_step < len(sim_close) else None
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
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
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
        st.info("Simulation prediction CSVs are not available on this server. The model artifacts were generated on a remote HPC cluster and the prediction files were not committed to the repository.")
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
            "Navigation", ["Overview","Stocks","Predictions","Simulation","Snapshots"],
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
    else:
        render_snapshots()


if __name__ == "__main__":
    main()
