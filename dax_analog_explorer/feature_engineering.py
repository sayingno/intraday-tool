"""Daily feature table -- one row per trading session.

All *analytical* quantities (gaps, distances, ranges, returns, ATR, regime) are
computed on the BACK-ADJUSTED price so nothing is contaminated by contract rolls.
Raw cash OHLC is also kept for display.  Every rolling / regime statistic uses
PRIOR sessions only (``.shift(1)`` before any rolling window) so a session's own
data can never leak into its own features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb
from . import ath_engine as ae


def _rolling_atr(high, low, close, window):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    # prior-only: shift the TR so the current day's TR is excluded
    return tr.shift(1).rolling(window, min_periods=max(3, window // 2)).mean()


def _volatility_regime(daily_ret: pd.Series, cfg: Config) -> pd.Series:
    """Low / normal / high regime from trailing realized vol, prior data only."""
    rv = daily_ret.shift(1).rolling(cfg.vol_window, min_periods=5).std()
    lo = rv.shift(1).rolling(cfg.regime_lookback, min_periods=20).quantile(cfg.regime_low_q)
    hi = rv.shift(1).rolling(cfg.regime_lookback, min_periods=20).quantile(cfg.regime_high_q)
    out = pd.Series(np.where(rv <= lo, "low",
                    np.where(rv >= hi, "high", "normal")), index=daily_ret.index)
    out[rv.isna() | lo.isna()] = "unknown"
    return out


def build_daily_features(master: pd.DataFrame, cfg: Config = DEFAULT_CONFIG,
                         calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    dohlc = sb.build_daily_ohlc(master, cfg)
    d = ae.compute_ath_table(master, dohlc, cfg).sort_values("session_date").reset_index(drop=True)

    # ---- previous-day levels (all adjusted) ----
    d["previous_close"] = d["cash_close_adj"].shift(1)
    d["previous_open"] = d["cash_open_adj"].shift(1)
    d["previous_day_high"] = d["cash_high_adj"].shift(1)
    d["previous_day_low"] = d["cash_low_adj"].shift(1)
    d["previous_day_close"] = d["cash_close_adj"].shift(1)
    d["previous_day_range"] = d["previous_day_high"] - d["previous_day_low"]
    d["previous_day_return"] = d["previous_close"] / d["cash_close_adj"].shift(2) - 1.0

    # ---- gap ----
    d["gap_points"] = d["cash_open_adj"] - d["previous_close"]
    d["gap_pct"] = 100.0 * d["gap_points"] / d["previous_close"]

    # ---- open vs previous range ----
    d["open_above_pdh"] = d["cash_open_adj"] > d["previous_day_high"]
    d["open_below_pdl"] = d["cash_open_adj"] < d["previous_day_low"]
    d["open_inside_previous_range"] = (
        (d["cash_open_adj"] >= d["previous_day_low"]) &
        (d["cash_open_adj"] <= d["previous_day_high"]))
    d["open_vs_pdh_pct"] = 100.0 * (d["cash_open_adj"] - d["previous_day_high"]) / d["previous_day_high"]
    d["open_vs_pdl_pct"] = 100.0 * (d["cash_open_adj"] - d["previous_day_low"]) / d["previous_day_low"]

    # ---- overnight ----
    d["overnight_return_pct"] = 100.0 * (d["overnight_close_adj"] - d["overnight_open_adj"]) / d["overnight_open_adj"]
    d["overnight_range"] = d["overnight_high_adj"] - d["overnight_low_adj"]
    onr = (d["overnight_high_adj"] - d["overnight_low_adj"]).replace(0, np.nan)
    d["open_location_in_overnight_range"] = (d["cash_open_adj"] - d["overnight_low_adj"]) / onr

    # ---- rolling ATR / range (prior-only) ----
    d["atr"] = _rolling_atr(d["cash_high_adj"], d["cash_low_adj"], d["cash_close_adj"], cfg.atr_window)
    d["rolling_daily_range"] = (d["cash_high_adj"] - d["cash_low_adj"]).shift(1).rolling(
        cfg.vol_window, min_periods=5).mean()
    d["overnight_range_normalized"] = d["overnight_range"] / d["atr"]
    d["gap_atr"] = d["gap_points"] / d["atr"]

    # ---- volatility regime (prior-only) ----
    daily_ret = d["cash_close_adj"].pct_change()
    d["daily_volatility_regime"] = _volatility_regime(daily_ret, cfg)

    # ---- data-quality flags ----
    if calendar is None:
        calendar = sb.build_calendar(sb.attach_sessions(master, cfg), cfg)
    cal = calendar.set_index("session_date")
    def quality(row):
        flags = []
        sd = row["session_date"]
        if sd in cal.index and bool(cal.loc[sd, "shortened_session"]):
            flags.append("shortened_session")
        if pd.isna(row["prior_ath"]):
            flags.append("no_prior_ath")
        if pd.isna(row["overnight_high_adj"]) or row.get("n_overnight_bars", 0) in (0, np.nan):
            flags.append("no_overnight_bars")
        if row["instrument"] == "FDAX5m":
            flags.append("fdax_source")
        return ";".join(flags)
    d["data_quality_flags"] = d.apply(quality, axis=1)

    # keep raw display OHLC under the plain spec names, adj under *_adj
    d = d.rename(columns={
        "cash_open": "cash_open_raw", "cash_high": "cash_high_raw",
        "cash_low": "cash_low_raw", "cash_close": "cash_close_raw",
    })
    d["cash_open"] = d["cash_open_adj"]
    d["cash_high"] = d["cash_high_adj"]
    d["cash_low"] = d["cash_low_adj"]
    d["cash_close"] = d["cash_close_adj"]
    return d


DAILY_FEATURE_COLUMNS = [
    "session_date", "weekday", "instrument",
    "cash_open", "cash_high", "cash_low", "cash_close",
    "cash_open_raw", "cash_high_raw", "cash_low_raw", "cash_close_raw",
    "previous_close", "previous_day_high", "previous_day_low", "previous_day_close",
    "previous_day_range", "previous_day_return",
    "prior_ath", "distance_open_to_ath_pct", "distance_overnight_high_to_ath_pct",
    "ath_at_open", "ath_reached_overnight",
    "ath_broken_first_5m", "ath_broken_first_15m", "ath_broken_first_30m",
    "gap_points", "gap_pct", "gap_atr",
    "open_above_pdh", "open_below_pdl", "open_inside_previous_range",
    "open_vs_pdh_pct", "open_vs_pdl_pct",
    "overnight_open_adj", "overnight_high_adj", "overnight_low_adj", "overnight_close_adj",
    "overnight_return_pct", "overnight_range", "overnight_range_normalized",
    "open_location_in_overnight_range",
    "atr", "rolling_daily_range", "daily_volatility_regime", "data_quality_flags",
]
