"""
Generate a PDF sanity report for the top-29 stocks.
Page 1: executive summary table
Pages 2-30: one page per symbol (4-panel chart + mini stats box)
"""

from __future__ import annotations
from pathlib import Path
import io

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, Image, PageBreak, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── paths ─────────────────────────────────────────────────────────────────────
CSV     = Path("reports/top29_eda/top29_15m_may2026.csv")
OUT_PDF = Path("reports/top29_eda/top29_sanity_report.pdf")
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

EXPECTED_TRADE_DATES = pd.bdate_range(
    "2026-05-01", "2026-05-30", freq="C", holidays=["2026-05-25"]
).date.tolist()
SLOTS_PER_DAY = 26

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV, parse_dates=["ts"])
df["ts"]         = pd.to_datetime(df["ts"])
df["trade_date"] = df["ts"].dt.date
df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
symbols = sorted(df["symbol"].unique())

# ── compute per-symbol stats ───────────────────────────────────────────────────
def symbol_stats(sym: str, sub: pd.DataFrame) -> dict:
    sub = sub.sort_values("ts").copy()
    sub["ret"] = sub["close"].pct_change()

    dates_present  = set(sub["trade_date"].unique())
    missing_dates  = sorted(set(EXPECTED_TRADE_DATES) - dates_present)
    nulls          = int(sub[["open","high","low","close","volume"]].isna().sum().sum())
    dups           = int(sub.duplicated(subset=["ts"]).sum())

    gaps = 0
    for _, grp in sub.groupby("trade_date"):
        deltas = grp.sort_values("ts")["ts"].diff().dt.total_seconds() / 60
        gaps  += int((deltas > 20).sum())

    ohlc_bad = int((
        (sub["high"] < sub["low"]) | (sub["high"] < sub["open"]) |
        (sub["high"] < sub["close"]) | (sub["low"] > sub["open"]) |
        (sub["low"] > sub["close"])
    ).sum())

    zero_vol = int((sub["volume"] <= 0).sum())

    stale, streak, prev = 0, 1, None
    for c in sub["close"]:
        if c == prev:
            streak += 1
            if streak == 5:
                stale += 1
        else:
            streak = 1
        prev = c

    outliers = int((sub["ret"].abs() > 0.05).dropna().sum())
    monthly_ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100

    status = "CLEAN" if not any([nulls, dups, gaps, ohlc_bad, missing_dates]) else "ISSUES"

    return dict(
        symbol=sym, status=status, bars=len(sub),
        missing_dates=len(missing_dates), intraday_gaps=gaps,
        nulls=nulls, duplicates=dups, ohlc_violations=ohlc_bad,
        zero_vol_bars=zero_vol, stale_events=stale, outlier_bars=outliers,
        monthly_return_pct=round(monthly_ret, 2),
        mean_close=round(sub["close"].mean(), 2),
        std_close=round(sub["close"].std(), 4),
        mean_volume=round(sub["volume"].mean(), 0),
    )

all_stats = [symbol_stats(s, df[df["symbol"] == s]) for s in symbols]
summary_df = pd.DataFrame(all_stats).sort_values("monthly_return_pct", ascending=False).reset_index(drop=True)

# ── chart helper ──────────────────────────────────────────────────────────────
def make_symbol_chart(sym: str, sub: pd.DataFrame) -> bytes:
    sub = sub.sort_values("ts").copy()
    sub["ret"]       = sub["close"].pct_change()
    sub["spread_pct"] = (sub["high"] - sub["low"]) / sub["close"] * 100

    daily = sub.groupby("trade_date").agg(
        close=("close", "last"), volume=("volume", "sum"), bars=("ts", "count")
    ).reset_index()

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # 1. close price
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(sub["ts"], sub["close"], linewidth=0.7, color="#2c7bb6")
    ax1.set_title("15-min Close Price", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Close ($)", fontsize=8)
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
    ax1.tick_params(labelsize=7)

    # 2. bars per day
    ax2 = fig.add_subplot(gs[1, 0])
    bar_colors = ["#e74c3c" if b < SLOTS_PER_DAY else "#2ecc71" for b in daily["bars"]]
    ax2.bar(range(len(daily)), daily["bars"], color=bar_colors)
    ax2.axhline(SLOTS_PER_DAY, color="black", linewidth=1, linestyle="--", label=f"exp {SLOTS_PER_DAY}")
    ax2.set_title("Bars per Trade Date", fontsize=9, fontweight="bold")
    ax2.set_ylabel("# bars", fontsize=8)
    ax2.set_xticks(range(len(daily)))
    ax2.set_xticklabels([str(d)[5:] for d in daily["trade_date"]], rotation=90, fontsize=5)
    ax2.legend(fontsize=6)

    # 3. daily volume
    med_vol = daily["volume"].median()
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(range(len(daily)), daily["volume"], color="#3498db", alpha=0.75)
    ax3.axhline(med_vol,      color="orange", linewidth=1.5, linestyle="--", label="median")
    ax3.axhline(med_vol * 3,  color="red",    linewidth=1,   linestyle=":",  label="3× spike")
    ax3.set_title("Daily Volume", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Volume", fontsize=8)
    ax3.set_xticks(range(len(daily)))
    ax3.set_xticklabels([str(d)[5:] for d in daily["trade_date"]], rotation=90, fontsize=5)
    ax3.legend(fontsize=6)
    ax3.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # 4. return distribution
    ax4 = fig.add_subplot(gs[2, 0])
    rets = sub["ret"].dropna()
    ax4.hist(rets, bins=60, color="#9b59b6", edgecolor="none", alpha=0.85)
    ax4.axvline(0,     color="black", linewidth=1)
    ax4.axvline( 0.05, color="red", linewidth=1, linestyle="--", label="±5%")
    ax4.axvline(-0.05, color="red", linewidth=1, linestyle="--")
    ax4.set_title("15-min Return Distribution", fontsize=9, fontweight="bold")
    ax4.set_xlabel("Return", fontsize=8)
    ax4.set_ylabel("Count", fontsize=8)
    ax4.tick_params(labelsize=7)
    ax4.legend(fontsize=6)

    # 5. bar spread %
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(sub["ts"], sub["spread_pct"], linewidth=0.5, color="#e67e22", alpha=0.75)
    ax5.set_title("Bar Spread % (H-L / Close)", fontsize=9, fontweight="bold")
    ax5.set_xlabel("Date", fontsize=8)
    ax5.set_ylabel("Spread %", fontsize=8)
    ax5.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
    ax5.tick_params(labelsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── build PDF ─────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=16, spaceAfter=6)
h1_style    = ParagraphStyle("H1",     parent=styles["Heading1"], fontSize=13, spaceAfter=4)
h2_style    = ParagraphStyle("H2",     parent=styles["Heading2"], fontSize=10, spaceAfter=3)
body_style  = ParagraphStyle("Body",   parent=styles["Normal"],   fontSize=8,  spaceAfter=3)
small_style = ParagraphStyle("Small",  parent=styles["Normal"],   fontSize=7,  textColor=colors.grey)
green = colors.HexColor("#27ae60")
red   = colors.HexColor("#e74c3c")
grey  = colors.HexColor("#ecf0f1")
dark  = colors.HexColor("#2c3e50")

doc   = SimpleDocTemplate(str(OUT_PDF), pagesize=A4,
                           leftMargin=1.5*cm, rightMargin=1.5*cm,
                           topMargin=1.5*cm, bottomMargin=1.5*cm)
story = []

# ── Page 1: Executive Summary ─────────────────────────────────────────────────
story.append(Paragraph("Top-29 Stock Data Sanity Report", title_style))
story.append(Paragraph("FMP 15-min OHLCV — May 2026", styles["Heading2"]))
story.append(Spacer(1, 0.3*cm))

# top-level stats
clean_count = (summary_df["status"] == "CLEAN").sum()
total_bars  = summary_df["bars"].sum()
total_out   = summary_df["outlier_bars"].sum()
total_stale = summary_df["stale_events"].sum()

meta_data = [
    ["Total symbols", str(len(symbols)),   "Trade dates", str(len(EXPECTED_TRADE_DATES))],
    ["Total bars",    f"{total_bars:,}",   "Bars / symbol", "520"],
    ["Clean symbols", f"{clean_count}/29", "Outlier bars",  str(total_out)],
    ["Stale events",  str(total_stale),    "Memorial Day",  "2026-05-26 skipped"],
]
meta_tbl = Table(meta_data, colWidths=[3.5*cm, 3*cm, 3.5*cm, 4.5*cm])
meta_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), grey),
    ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
    ("FONTSIZE",   (0,0), (-1,-1), 8),
    ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTNAME",   (2,0), (2,-1), "Helvetica-Bold"),
    ("GRID",       (0,0), (-1,-1), 0.25, colors.white),
    ("ROWBACKGROUNDS", (0,0), (-1,-1), [grey, colors.white]),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING",    (0,0), (-1,-1), 4),
]))
story.append(meta_tbl)
story.append(Spacer(1, 0.4*cm))
story.append(HRFlowable(width="100%", thickness=1, color=dark))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Per-Symbol Sanity Summary", h1_style))

# summary table header
tbl_header = [
    "Symbol", "Status", "Bars", "Miss\nDates", "ID\nGaps",
    "Nulls", "Dups", "OHLC\nViol", "0-Vol", "Stale", "Outlier\nBars", "Return %"
]
tbl_rows = [tbl_header]
for _, r in summary_df.iterrows():
    ret_str = f"{r['monthly_return_pct']:+.2f}%"
    tbl_rows.append([
        r["symbol"], r["status"], str(r["bars"]),
        str(r["missing_dates"]), str(r["intraday_gaps"]),
        str(r["nulls"]), str(r["duplicates"]), str(r["ohlc_violations"]),
        str(r["zero_vol_bars"]), str(r["stale_events"]), str(r["outlier_bars"]),
        ret_str,
    ])

col_w = [1.4*cm, 1.4*cm, 1.0*cm, 1.0*cm, 0.9*cm,
         0.9*cm, 0.8*cm, 1.0*cm, 0.8*cm, 0.9*cm, 1.1*cm, 1.5*cm]
summary_tbl = Table(tbl_rows, colWidths=col_w, repeatRows=1)

tbl_style = [
    ("BACKGROUND",    (0, 0), (-1, 0),  dark),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 7),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("GRID",          (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, grey]),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING",    (0, 0), (-1, -1), 3),
]
# colour status & return cells
for i, (_, r) in enumerate(summary_df.iterrows(), start=1):
    if r["status"] == "CLEAN":
        tbl_style.append(("TEXTCOLOR", (1, i), (1, i), green))
        tbl_style.append(("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"))
    else:
        tbl_style.append(("BACKGROUND", (1, i), (1, i), colors.HexColor("#fadbd8")))
    ret = r["monthly_return_pct"]
    ret_col = green if ret >= 0 else red
    tbl_style.append(("TEXTCOLOR", (11, i), (11, i), ret_col))
    tbl_style.append(("FONTNAME",  (11, i), (11, i), "Helvetica-Bold"))

summary_tbl.setStyle(TableStyle(tbl_style))
story.append(summary_tbl)

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph(
    "<b>Notes:</b>  ID Gaps = intraday gaps &gt;20 min within a trade day. "
    "Outlier Bars = |15-min return| &gt; 5% (normal at open/earnings). "
    "Stale = same close repeated ≥5 consecutive bars.",
    small_style,
))
story.append(PageBreak())

# ── Pages 2-30: per-symbol ────────────────────────────────────────────────────
print("Building per-symbol pages ...")
for i, row in summary_df.iterrows():
    sym  = row["symbol"]
    sub  = df[df["symbol"] == sym]
    stat = dict(row)

    print(f"  {sym}")

    story.append(Paragraph(f"{sym} — Sanity Detail", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 0.2*cm))

    # mini stats row
    ret_color = "#27ae60" if stat["monthly_return_pct"] >= 0 else "#e74c3c"
    status_color = "#27ae60" if stat["status"] == "CLEAN" else "#e74c3c"
    mini = [
        ["Status", "Bars", "Monthly Return", "Mean Close", "Std Close", "Mean Volume"],
        [
            Paragraph(f'<font color="{status_color}"><b>{stat["status"]}</b></font>', body_style),
            str(stat["bars"]),
            Paragraph(f'<font color="{ret_color}"><b>{stat["monthly_return_pct"]:+.2f}%</b></font>', body_style),
            f"${stat['mean_close']:,.2f}",
            f"{stat['std_close']:.4f}",
            f"{int(stat['mean_volume']):,}",
        ],
    ]
    mini_tbl = Table(mini, colWidths=[2.2*cm]*6)
    mini_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), dark),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("GRID",          (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BACKGROUND",    (0,1), (-1,1), grey),
    ]))
    story.append(mini_tbl)
    story.append(Spacer(1, 0.15*cm))

    # issue flags row
    checks = [
        ("Nulls",        stat["nulls"],          0),
        ("Duplicates",   stat["duplicates"],      0),
        ("Missing Dates",stat["missing_dates"],   0),
        ("ID Gaps",      stat["intraday_gaps"],   0),
        ("OHLC Viol",    stat["ohlc_violations"], 0),
        ("Zero-Vol Bars",stat["zero_vol_bars"],   0),
        ("Stale Events", stat["stale_events"],    0),
        ("Outlier Bars", stat["outlier_bars"],    0),
    ]
    flag_header = [c[0] for c in checks]
    flag_vals   = []
    flag_style  = [
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("GRID",          (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#bdc3c7")),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
    ]
    row_vals = []
    for j, (label, val, threshold) in enumerate(checks):
        is_bad = val > threshold
        cell   = Paragraph(
            f'<font color="{"#e74c3c" if is_bad else "#27ae60"}"><b>{val}</b></font>',
            body_style
        )
        row_vals.append(cell)
        if is_bad:
            flag_style.append(("BACKGROUND", (j,1), (j,1), colors.HexColor("#fadbd8")))
    flag_vals.append(row_vals)
    flag_tbl = Table([flag_header, *flag_vals], colWidths=[2.0*cm]*8)
    flag_tbl.setStyle(TableStyle(flag_style))
    story.append(flag_tbl)
    story.append(Spacer(1, 0.3*cm))

    # chart
    chart_bytes = make_symbol_chart(sym, sub)
    img = Image(io.BytesIO(chart_bytes), width=17*cm, height=11.7*cm)
    story.append(img)

    if i < len(summary_df) - 1:
        story.append(PageBreak())

# ── build ─────────────────────────────────────────────────────────────────────
print("Building PDF ...")
doc.build(story)
print(f"\nSaved: {OUT_PDF}  ({OUT_PDF.stat().st_size/1024:.0f} KB)")
