"""Render matching sessions to a multi-page PDF of annotated candlestick charts.

Each page is one session: 5-minute candles for the cash session, the reference
levels (PDH / PDC / PDL / overnight high & low / the day's open), the opening
range, a marker at the decision bar, and -- when trade details are supplied --
the entry, stop, target and where the trade actually ended.

The observed window (open -> decision bar) is shaded differently from the
outcome window so it is obvious at a glance what the signal could see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e6e2"
UP = "#2e9e5b"
DOWN = "#d1495b"
BLUE = "#2a78d6"
OBS_FILL = "#2a78d6"
LEVELS = [("PDH", "previous_day_high", "#c77b2a"),
          ("PDC", "previous_close", "#8a8984"),
          ("PDL", "previous_day_low", "#c77b2a"),
          ("ONH", "overnight_high_adj", "#b06bb0"),
          ("ONL", "overnight_low_adj", "#b06bb0")]


def _candles(ax, b: pd.DataFrame, width: float = 3.4) -> None:
    for _, r in b.iterrows():
        up = r["adj_close"] >= r["adj_open"]
        col = UP if up else DOWN
        ax.vlines(r["mso"], r["adj_low"], r["adj_high"], color=col, lw=0.9, zorder=3)
        lo = min(r["adj_open"], r["adj_close"])
        h = abs(r["adj_close"] - r["adj_open"])
        ax.add_patch(Rectangle((r["mso"] - width / 2, lo), width, max(h, 1e-9),
                               facecolor=col, edgecolor=col, lw=0.6, zorder=4))


def draw_session(ax, bars: pd.DataFrame, day: pd.Series, decision_min: int,
                 trade: dict | None = None, title: str = "") -> None:
    b = bars.sort_values("mso")
    _candles(ax, b)

    lo_all, hi_all = float(b["adj_low"].min()), float(b["adj_high"].max())
    pad = (hi_all - lo_all) * 0.10 or 1.0

    # observed vs outcome shading
    ax.axvspan(b["mso"].min() - 3, decision_min, color=OBS_FILL, alpha=0.07, lw=0, zorder=1)
    ax.axvline(decision_min, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=6)

    # opening range
    obs = b[b["mso"] < decision_min]
    if len(obs):
        or_hi, or_lo = float(obs["adj_high"].max()), float(obs["adj_low"].min())
        ax.add_patch(Rectangle((b["mso"].min() - 3, or_lo),
                               decision_min - b["mso"].min() + 3, or_hi - or_lo,
                               facecolor="none", edgecolor=BLUE, lw=1.0,
                               ls=(0, (3, 2)), zorder=5))

    # reference levels
    xmax = float(b["mso"].max())
    for name, col, colour in LEVELS:
        v = day.get(col)
        if v is None or pd.isna(v) or not (lo_all - pad <= v <= hi_all + pad):
            continue
        ax.axhline(v, color=colour, lw=0.9, ls=(0, (5, 3)), zorder=2)
        ax.text(xmax + 4, v, f" {name}", va="center", fontsize=7, color=colour, clip_on=False)
    o = day.get("cash_open_adj")
    if o is not None and not pd.isna(o):
        ax.axhline(o, color=INK_2, lw=1.0, zorder=2)
        ax.text(xmax + 4, o, " OPEN", va="center", fontsize=7, color=INK_2, clip_on=False)

    # trade overlay
    if trade:
        ax.axhline(trade["entry"], color=BLUE, lw=1.2, zorder=7)
        ax.text(xmax + 4, trade["entry"], " entry", va="center", fontsize=7,
                color=BLUE, clip_on=False)
        ax.axhline(trade["stop_init"], color=DOWN, lw=1.0, ls=(0, (2, 2)), zorder=7)
        ax.text(xmax + 4, trade["stop_init"], " stop", va="center", fontsize=7,
                color=DOWN, clip_on=False)
        ax.plot([trade["exit_min"]], [trade["exit_price"]], marker="o", ms=6,
                color=(UP if trade["R_multiple"] > 0 else DOWN), zorder=8)
        ax.annotate(f"{trade['reason']}  {trade['R_multiple']:+.2f}R",
                    (trade["exit_min"], trade["exit_price"]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8,
                    color=(UP if trade["R_multiple"] > 0 else DOWN))

    ax.set_xlim(b["mso"].min() - 6, xmax + 6)
    ax.set_ylim(lo_all - pad, hi_all + pad)
    ticks = [t for t in (0, 60, 120, 180, 240, 300, 360, 420, 480) if t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{9 + t // 60:02d}:{t % 60:02d}" for t in ticks],
                       fontsize=8, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def multiday_frame(master: pd.DataFrame, date, lookback: int = 2,
                   cfg: Config = DEFAULT_CONFIG, ema_fast: int = 20,
                   ema_slow: int = 50) -> pd.DataFrame:
    """Cash-session bars for ``date`` plus the ``lookback`` sessions before it,
    concatenated on a continuous index so the overnight void is collapsed --
    the same "Xetra trading hours" view a charting package shows.

    The moving averages are computed on that continuous cash-only series, so they
    match what is on screen rather than being contaminated by overnight bars.
    """
    ms = sb.attach_sessions(master, cfg)
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0)].sort_values("dt")
    days = sorted(cash["session_date"].unique())
    date = pd.Timestamp(date).normalize()
    if date not in days:
        return pd.DataFrame()
    i = days.index(date)
    window = days[max(0, i - lookback): i + 1]
    f = cash[cash["session_date"].isin(window)].copy().sort_values("dt").reset_index(drop=True)
    f["x"] = np.arange(len(f))
    f["ema_fast"] = f["adj_close"].ewm(span=ema_fast, adjust=False).mean()
    f["ema_slow"] = f["adj_close"].ewm(span=ema_slow, adjust=False).mean()
    return f


def draw_multiday(ax, f: pd.DataFrame, signal_date, day: pd.Series,
                  decision_min: int, trade: dict | None = None, title: str = "") -> None:
    """Multi-session panel: candles, both EMAs, session separators, the signal
    session shaded, and its levels drawn only across that session."""
    for _, r in f.iterrows():
        up = r["adj_close"] >= r["adj_open"]
        col = UP if up else DOWN
        ax.vlines(r["x"], r["adj_low"], r["adj_high"], color=col, lw=0.7, zorder=3)
        lo = min(r["adj_open"], r["adj_close"])
        h = abs(r["adj_close"] - r["adj_open"])
        ax.add_patch(Rectangle((r["x"] - 0.36, lo), 0.72, max(h, 1e-9),
                               facecolor=col, edgecolor=col, lw=0.4, zorder=4))
    ax.plot(f["x"], f["ema_fast"], color=BLUE, lw=1.1, zorder=5)
    ax.plot(f["x"], f["ema_slow"], color="#d08a8a", lw=1.1, zorder=5)

    sig = pd.Timestamp(signal_date).normalize()
    sd = f[f["session_date"] == sig]
    # session separators + date labels
    for d, grp in f.groupby("session_date"):
        x0 = float(grp["x"].min())
        ax.axvline(x0 - 0.5, color=MUTED, lw=0.8, zorder=2)
        ax.text(x0 + 1, ax.get_ylim()[0], "", fontsize=7)
    if len(sd):
        x0, x1 = float(sd["x"].min()), float(sd["x"].max())
        ax.axvspan(x0 - 0.5, x1 + 0.5, color="#f0c07a", alpha=0.10, lw=0, zorder=1)
        dec_x = sd[sd["mso"] <= decision_min]["x"]
        if len(dec_x):
            ax.axvline(float(dec_x.max()) + 0.5, color=INK, lw=1.3,
                       ls=(0, (4, 3)), zorder=6)
        obs = sd[sd["mso"] < decision_min]
        if len(obs):
            ax.add_patch(Rectangle((x0 - 0.5, float(obs["adj_low"].min())),
                                   float(obs["x"].max()) - x0 + 1,
                                   float(obs["adj_high"].max() - obs["adj_low"].min()),
                                   facecolor="none", edgecolor=BLUE, lw=1.0,
                                   ls=(0, (3, 2)), zorder=5))
        # levels, drawn only across the signal session
        for name, col, colour in LEVELS:
            v = day.get(col)
            if v is None or pd.isna(v):
                continue
            ax.hlines(v, x0 - 0.5, x1 + 0.5, color=colour, lw=0.9,
                      ls=(0, (5, 3)), zorder=2)
            ax.text(x1 + 1.5, v, f" {name}", va="center", fontsize=7,
                    color=colour, clip_on=False)
        if trade:
            ax.hlines(trade["entry"], x0, x1 + 0.5, color=BLUE, lw=1.2, zorder=7)
            ax.hlines(trade["stop_init"], x0, x1 + 0.5, color=DOWN, lw=1.0,
                      ls=(0, (2, 2)), zorder=7)

    # x labels: one per session
    ticks, labels = [], []
    for d, grp in f.groupby("session_date"):
        ticks.append(float(grp["x"].min()) + len(grp) / 2)
        labels.append(pd.Timestamp(d).strftime("%d %b"))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8, color=INK_2)
    ax.set_xlim(-1, float(f["x"].max()) + 8)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")


def multiday_pdf(path: str, dates, master: pd.DataFrame, daily: pd.DataFrame,
                 decision_min: int = 30, lookback: int = 2,
                 trades: pd.DataFrame | None = None, context: pd.DataFrame | None = None,
                 cfg: Config = DEFAULT_CONFIG, per_page: int = 2,
                 title: str = "Sessions in context", subtitle: str = "") -> str:
    di = daily.set_index("session_date")
    tr = trades.set_index("session_date") if trades is not None and len(trades) else None
    cx = context.set_index("session_date") if context is not None and len(context) else None
    dates = [pd.Timestamp(d) for d in dates]

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3), facecolor=SURFACE)
        fig.text(0.06, 0.86, title, fontsize=21, color=INK, weight="bold")
        fig.text(0.06, 0.805, subtitle or f"{len(dates)} sessions, each with the "
                 f"{lookback} sessions before it · Xetra hours only (09:00–17:30)",
                 fontsize=11, color=INK_2)
        for i, t in enumerate([
            "Shaded session   the signal day; earlier sessions are context",
            "Blue / red line  EMA 20 and EMA 50 on cash-session bars only",
            "Dashed box       opening range;  dashed vertical  the decision bar",
            "PDH/PDC/PDL      previous day high / close / low     ONH/ONL  overnight high / low",
        ]):
            fig.text(0.06, 0.72 - i * 0.035, t, fontsize=9.5, color=INK_2)
        fig.text(0.06, 0.09, "Historical record, not a recommendation.",
                 fontsize=8.5, color=MUTED)
        pdf.savefig(fig, facecolor=SURFACE); plt.close(fig)

        for i in range(0, len(dates), per_page):
            chunk = dates[i:i + per_page]
            fig, axes = plt.subplots(per_page, 1, figsize=(11.7, 8.3),
                                     facecolor=SURFACE, squeeze=False)
            for ax, d in zip(axes[:, 0], chunk):
                ax.set_facecolor(SURFACE)
                f = multiday_frame(master, d, lookback, cfg)
                if not len(f):
                    ax.axis("off"); continue
                day = di.loc[d] if d in di.index else pd.Series(dtype=float)
                bits = [str(pd.Timestamp(d).date()), pd.Timestamp(d).day_name()[:3]]
                if cx is not None and d in cx.index:
                    c = cx.loc[d]
                    bits += [str(c.get("opening")), str(c.get("location")),
                             f"gap {c.get('gap_bucket')}", f"prior {c.get('prior_day_type')}"]
                t = tr.loc[d].to_dict() if (tr is not None and d in tr.index) else None
                if t:
                    bits.append(f"{t['reason']} {t['R_multiple']:+.2f}R")
                draw_multiday(ax, f, d, day, decision_min, t, "   ·   ".join(bits))
            for ax in axes[len(chunk):, 0]:
                ax.axis("off")
            fig.tight_layout(rect=(0.01, 0.01, 0.97, 0.99))
            pdf.savefig(fig, facecolor=SURFACE); plt.close(fig)
    return path


def sessions_to_pdf(path: str, dates, master: pd.DataFrame, daily: pd.DataFrame,
                    decision_min: int = 30, trades: pd.DataFrame | None = None,
                    context: pd.DataFrame | None = None,
                    cfg: Config = DEFAULT_CONFIG, per_page: int = 2,
                    title: str = "Matching sessions") -> str:
    ms = sb.attach_sessions(master, cfg)
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0)]
    groups = dict(tuple(cash.groupby("session_date")))
    di = daily.set_index("session_date")
    tr = trades.set_index("session_date") if trades is not None and len(trades) else None
    cx = context.set_index("session_date") if context is not None and len(context) else None

    dates = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) in groups]
    with PdfPages(path) as pdf:
        # cover
        fig = plt.figure(figsize=(11.7, 8.3), facecolor=SURFACE)
        fig.text(0.06, 0.86, title, fontsize=22, color=INK, weight="bold")
        fig.text(0.06, 0.80, f"{len(dates)} sessions · decision bar at "
                             f"{decision_min} min after the 09:00 open "
                             f"(={9 + decision_min // 60:02d}:{decision_min % 60:02d} Berlin)",
                 fontsize=11, color=INK_2)
        legend = [
            "Shaded band  observed window — everything the signal was allowed to see",
            "Dashed box   opening range (high/low of the observed window)",
            "Dashed line  decision bar; entries fill on the NEXT bar's open",
            "PDH/PDC/PDL  previous day high / close / low     ONH/ONL  overnight high / low",
            "Marker       where the trade ended, with its R multiple",
        ]
        for i, t in enumerate(legend):
            fig.text(0.06, 0.70 - i * 0.035, t, fontsize=9.5, color=INK_2)
        fig.text(0.06, 0.10, "Historical record, not a recommendation. Sample sizes are "
                             "shown wherever a rate is quoted.", fontsize=8.5, color=MUTED)
        pdf.savefig(fig, facecolor=SURFACE); plt.close(fig)

        for i in range(0, len(dates), per_page):
            chunk = dates[i:i + per_page]
            fig, axes = plt.subplots(per_page, 1, figsize=(11.7, 8.3),
                                     facecolor=SURFACE, squeeze=False)
            for ax, d in zip(axes[:, 0], chunk):
                ax.set_facecolor(SURFACE)
                day = di.loc[d] if d in di.index else pd.Series(dtype=float)
                bits = [str(pd.Timestamp(d).date()), pd.Timestamp(d).day_name()[:3]]
                if cx is not None and d in cx.index:
                    c = cx.loc[d]
                    bits += [str(c.get("opening")), str(c.get("location")),
                             str(c.get("swing")), f"gap {c.get('gap_bucket')}"]
                t = None
                if tr is not None and d in tr.index:
                    t = tr.loc[d].to_dict()
                    bits.append(f"{t['reason']} {t['R_multiple']:+.2f}R")
                draw_session(ax, groups[d], day, decision_min, t, "   ·   ".join(bits))
            for ax in axes[len(chunk):, 0]:
                ax.axis("off")
            fig.tight_layout(rect=(0.01, 0.01, 0.96, 0.99))
            pdf.savefig(fig, facecolor=SURFACE); plt.close(fig)
    return path
