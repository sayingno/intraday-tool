"""Plotly charts.

The one non-negotiable: every intraday chart visually separates the OBSERVED
window (09:00 .. cutoff) from the OUTCOME window (after the cutoff) with a
vertical marker and shading, so it is obvious what the matcher was allowed to see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import Config, DEFAULT_CONFIG

OBS_COLOR = "rgba(70,130,180,0.10)"
OUT_COLOR = "rgba(200,120,60,0.10)"


def day_candles(bars: pd.DataFrame, cutoff_min: int, cfg: Config = DEFAULT_CONFIG,
                title: str = "", use_adj: bool = False) -> go.Figure:
    """Candlestick for one session with observed/outcome split at the cutoff."""
    b = bars.sort_values("mso")
    pre = "adj_" if use_adj else ""
    fig = go.Figure(go.Candlestick(
        x=b["mso"], open=b[f"{pre}open"], high=b[f"{pre}high"],
        low=b[f"{pre}low"], close=b[f"{pre}close"], name="price",
        increasing_line_color="#2e9e5b", decreasing_line_color="#d1495b"))
    lo = float(b[f"{pre}low"].min()); hi = float(b[f"{pre}high"].max())
    fig.add_vrect(x0=b["mso"].min(), x1=cutoff_min, fillcolor=OBS_COLOR,
                  line_width=0, annotation_text="observed", annotation_position="top left")
    fig.add_vrect(x0=cutoff_min, x1=b["mso"].max(), fillcolor=OUT_COLOR,
                  line_width=0, annotation_text="outcome", annotation_position="top right")
    fig.add_vline(x=cutoff_min, line_dash="dash", line_color="#444")
    fig.update_layout(title=title, xaxis_title="minutes since 09:00 (Berlin)",
                      yaxis_title="price", xaxis_rangeslider_visible=False,
                      height=360, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def analog_overlay(grid: np.ndarray, mat: np.ndarray, kept: list, ref_date,
                   cutoff_min: int, unit: str = "atr", max_lines: int = 25) -> go.Figure:
    """All aligned paths (open=0), reference highlighted, cutoff marked."""
    fig = go.Figure()
    for i, d in enumerate(kept):
        if d == ref_date:
            continue
        if i >= max_lines:
            break
        fig.add_trace(go.Scatter(x=grid, y=mat[i], mode="lines",
                      line=dict(width=1, color="rgba(120,120,120,0.35)"),
                      name=str(pd.Timestamp(d).date()), showlegend=False))
    if ref_date in kept:
        ri = kept.index(ref_date)
        fig.add_trace(go.Scatter(x=grid, y=mat[ri], mode="lines",
                      line=dict(width=3, color="#1f77b4"),
                      name=f"REF {pd.Timestamp(ref_date).date()}"))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.add_vline(x=cutoff_min, line_dash="dash", line_color="#444")
    fig.update_layout(title="Aligned opening paths (rebased at cash open)",
                      xaxis_title="minutes since 09:00", yaxis_title=f"move ({unit})",
                      height=380, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def median_band(grid: np.ndarray, mat: np.ndarray, cutoff_min: int,
                ref: np.ndarray | None = None, unit: str = "atr") -> go.Figure:
    """Median analog path with a 25-75 percentile band."""
    if mat.shape[0] == 0:
        return go.Figure()
    med = np.nanmedian(mat, axis=0)
    q1 = np.nanpercentile(mat, 25, axis=0)
    q3 = np.nanpercentile(mat, 75, axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([grid, grid[::-1]]),
                  y=np.concatenate([q3, q1[::-1]]), fill="toself",
                  fillcolor="rgba(31,119,180,0.15)", line=dict(width=0),
                  name="25-75 pctile"))
    fig.add_trace(go.Scatter(x=grid, y=med, mode="lines",
                  line=dict(width=3, color="#1f77b4"), name="median analog"))
    if ref is not None:
        fig.add_trace(go.Scatter(x=grid, y=ref, mode="lines",
                      line=dict(width=2, dash="dot", color="#d1495b"), name="reference"))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.add_vline(x=cutoff_min, line_dash="dash", line_color="#444")
    fig.update_layout(title="Median analog path + 25-75 band",
                      xaxis_title="minutes since 09:00", yaxis_title=f"move ({unit})",
                      height=380, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def outcome_distribution(values: np.ndarray, title: str, xlabel: str) -> go.Figure:
    v = np.asarray(values, dtype="float64")
    v = v[~np.isnan(v)]
    fig = go.Figure(go.Histogram(x=v, nbinsx=25, marker_color="#1f77b4"))
    if len(v):
        fig.add_vline(x=float(np.median(v)), line_dash="dash", line_color="#d1495b",
                      annotation_text=f"median {np.median(v):.2f}")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="days",
                      height=300, margin=dict(l=40, r=20, t=40, b=40))
    return fig


OTC_INK = "#1f4e79"
LEVEL_COLORS = {"PDH": "#c77b2a", "PDC": "#8a8984", "PDL": "#c77b2a",
                "ONH": "#b06bb0", "ONL": "#b06bb0", "OPEN": "#52514e"}


def otc_session(bars: pd.DataFrame, levels: dict, mark: dict | None,
                deadline_min: int, title: str = "") -> go.Figure:
    """One session with the OTC anatomy drawn on it.

    Same contract as the PDF overlay in ``chart_pdf``: every line is a value the
    engine decided on, looked up through the bar's ``mso``, never re-derived from
    the prices.  The x axis is minutes since 09:00 so it lines up with everything
    else the tool prints.
    """
    b = bars.sort_values("mso")
    fig = go.Figure(go.Candlestick(
        x=b["mso"], open=b["adj_open"], high=b["adj_high"], low=b["adj_low"],
        close=b["adj_close"], name="price", showlegend=False,
        increasing_line_color="#2e9e5b", decreasing_line_color="#d1495b"))
    lo, hi = float(b["adj_low"].min()), float(b["adj_high"].max())
    pad = (hi - lo) * 0.08 or 1.0
    xmax = float(b["mso"].max())

    for name, v in levels.items():
        if v is None or pd.isna(v) or not (lo - pad <= v <= hi + pad):
            continue
        fig.add_hline(y=float(v), line_dash="dot", line_width=1,
                      line_color=LEVEL_COLORS.get(name, "#999"),
                      annotation_text=name, annotation_position="right",
                      annotation_font_size=10)

    fig.add_vrect(x0=b["mso"].min(), x1=deadline_min, fillcolor=OBS_COLOR, line_width=0)
    fig.add_vline(x=deadline_min, line_dash="dash", line_color="#444",
                  annotation_text="trigger deadline", annotation_position="top left",
                  annotation_font_size=10)

    if mark:
        xof = dict(zip(b["mso"].astype(int), b["mso"].astype(float)))
        drawn = set()
        for s in mark.get("signals", []):
            up = int(s.get("side", 1)) > 0
            taken = bool(s.get("taken", True))
            status = s.get("status")
            live = taken and status in (None, "TRADED")
            col = OTC_INK if live else "#8a8984"

            key = (s.get("leg_origin_px"), s.get("leg_extreme_px"), up)
            if s.get("leg_origin_px") is not None and key not in drawn:
                drawn.add(key)
                fig.add_hline(y=float(s["leg_origin_px"]), line_width=1.4,
                              line_color=OTC_INK, annotation_text="leg origin",
                              annotation_position="left", annotation_font_size=10)
                if s.get("leg_extreme_mso") is not None:
                    fig.add_trace(go.Scatter(
                        x=[s["leg_extreme_mso"]], y=[s["leg_extreme_px"]], mode="markers",
                        marker=dict(symbol="triangle-down" if up else "triangle-up",
                                    size=11, color=OTC_INK),
                        name="leg high" if up else "leg low", showlegend=False,
                        hovertemplate=("leg high" if up else "leg low") + " %{y:.0f}<extra></extra>"))

            sx = xof.get(int(s["signal_mso"]))
            if sx is None:
                continue
            xe = s.get("exit_mso") if s.get("exit_mso") is not None else min(xmax, sx + 60)
            note = ("" if live else
                    " (cancelled)" if status == "CANCELLED" else
                    " (never triggered)" if status == "NOT_TRIGGERED" else " (not taken)")
            fig.add_trace(go.Scatter(
                x=[sx], y=[s["entry_px"]], mode="markers+text",
                marker=dict(symbol="triangle-up" if up else "triangle-down",
                            size=13, color=col),
                text=[s["label"] + note], textposition="top center" if up else "bottom center",
                textfont=dict(size=11, color=col), showlegend=False,
                hovertemplate=f"{s['label']} trigger %{{y:.0f}}<extra></extra>"))
            for y, colour, dash in ((s["entry_px"], col, "solid"),
                                    (s["stop_px"], "#d1495b", "dash"),
                                    (s.get("target_px"), "#2e9e5b", "dashdot")):
                if y is None or pd.isna(y):
                    continue
                fig.add_shape(type="line", x0=sx, x1=xe, y0=float(y), y1=float(y),
                              line=dict(color=colour, width=1.4, dash=dash))
            if s.get("exit_mso") is not None and s.get("exit_px") is not None:
                good = (s.get("R_multiple") or 0) > 0
                fig.add_trace(go.Scatter(
                    x=[s["exit_mso"]], y=[s["exit_px"]], mode="markers+text",
                    marker=dict(size=9, color="#2e9e5b" if good else "#d1495b"),
                    text=[f"{s.get('reason','')} {s.get('R_multiple',0):+.2f}R"],
                    textposition="middle right", textfont=dict(size=10),
                    showlegend=False, hoverinfo="skip"))

    ticks = list(range(0, int(xmax) + 1, 30))
    fig.update_layout(title=title, xaxis_title="Berlin time",
                      yaxis_title="price (back-adjusted)",
                      xaxis_rangeslider_visible=False, height=520,
                      margin=dict(l=40, r=60, t=50, b=40),
                      xaxis=dict(tickmode="array", tickvals=ticks,
                                 ticktext=[f"{9 + t // 60:02d}:{t % 60:02d}" for t in ticks]))
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig
