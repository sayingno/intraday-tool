"""Outcome engine -- what happened AFTER the observation cutoff.

This is the only place allowed to read post-cutoff bars, because outcomes are the
*answer*, never an input to matching.  Everything is on adjusted prices and keyed
off the price at the cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb


def _price_at(bars: pd.DataFrame, target_mso: int):
    """Close of the last bar at or before target minutes-since-open."""
    b = bars[bars["mso"] <= target_mso]
    if len(b) == 0:
        return np.nan
    return float(b.sort_values("mso")["adj_close"].iloc[-1])


def _touched(bars_after: pd.DataFrame, level: float) -> bool:
    if pd.isna(level) or len(bars_after) == 0:
        return False
    return bool((bars_after["adj_low"] <= level).any() and (bars_after["adj_high"] >= level).any() or
                (bars_after["adj_high"] >= level).any() and (bars_after["adj_low"] <= level).any())


def _crossed_up(bars_after, level):
    return bool(len(bars_after) and not pd.isna(level) and (bars_after["adj_high"] >= level).any())


def _crossed_down(bars_after, level):
    return bool(len(bars_after) and not pd.isna(level) and (bars_after["adj_low"] <= level).any())


def session_outcome(cash_bars: pd.DataFrame, ref: dict, cutoff_min: int,
                    cfg: Config = DEFAULT_CONFIG) -> dict:
    """Post-cutoff outcome dict for one session (cash_bars = full cash session)."""
    b = cash_bars.sort_values("mso")
    obs = b[b["mso"] <= cutoff_min]
    after = b[b["mso"] > cutoff_min]
    out = {"cutoff_min": cutoff_min}
    if len(obs) == 0:
        return out
    cutoff_price = float(obs["adj_close"].iloc[-1])
    out["cutoff_price"] = cutoff_price
    open_px = ref["cash_open"]

    # returns at horizons
    for hmin in cfg.outcome_horizons_min:
        p = _price_at(b, cutoff_min + hmin)
        out[f"return_after_{hmin}m_pct"] = 100.0 * (p - cutoff_price) / cutoff_price if not pd.isna(p) else np.nan
    for name, tmin in (("1200", 180), ("us_open", cfg.minutes_since_open(cfg.us_cash_open)),
                       ("cash_close", cfg.minutes_since_open(cfg.cash_close))):
        p = _price_at(b, tmin)
        out[f"return_at_{name}_pct"] = 100.0 * (p - cutoff_price) / cutoff_price if not pd.isna(p) else np.nan

    # MFE / MAE from the cutoff price (over the post-cutoff remainder)
    if len(after):
        out["max_favorable_excursion_pct"] = 100.0 * (after["adj_high"].max() - cutoff_price) / cutoff_price
        out["max_adverse_excursion_pct"] = 100.0 * (after["adj_low"].min() - cutoff_price) / cutoff_price
    else:
        out["max_favorable_excursion_pct"] = np.nan
        out["max_adverse_excursion_pct"] = np.nan

    # session high / low time (full cash session)
    hi_i = b["adj_high"].idxmax(); lo_i = b["adj_low"].idxmin()
    out["session_high_min"] = float(b.loc[hi_i, "mso"])
    out["session_low_min"] = float(b.loc[lo_i, "mso"])
    out["session_close_pct_vs_cutoff"] = 100.0 * (float(b["adj_close"].iloc[-1]) - cutoff_price) / cutoff_price

    # observed-session (up to cutoff) high/low breaks after cutoff
    obs_high = obs["adj_high"].max(); obs_low = obs["adj_low"].min()
    out["break_observed_high"] = _crossed_up(after, obs_high)
    out["break_observed_low"] = _crossed_down(after, obs_low)
    out["retest_cash_open"] = _touched(after, open_px)

    # touches of key levels after cutoff
    out["touch_overnight_high"] = _crossed_up(after, ref.get("overnight_high"))
    out["touch_overnight_low"] = _crossed_down(after, ref.get("overnight_low"))
    out["touch_pdh"] = _crossed_up(after, ref.get("pdh"))
    out["touch_pdl"] = _crossed_down(after, ref.get("pdl"))
    out["touch_previous_close"] = _touched(after, ref.get("prev_close"))

    # gap fill (full = reaches previous cash close; half = halfway there)
    prev_close = ref.get("prev_close")
    if not pd.isna(prev_close):
        half = (open_px + prev_close) / 2.0
        gap_up = open_px > prev_close
        out["full_gap_fill"] = _crossed_down(after, prev_close) if gap_up else _crossed_up(after, prev_close)
        out["half_gap_fill"] = _crossed_down(after, half) if gap_up else _crossed_up(after, half)
    else:
        out["full_gap_fill"] = out["half_gap_fill"] = False

    # end-of-session relations
    close_px = float(b["adj_close"].iloc[-1])
    out["close_above_cash_open"] = bool(close_px > open_px)
    out["close_below_cash_open"] = bool(close_px < open_px)
    if not pd.isna(ref.get("prior_ath")):
        out["close_above_prior_ath"] = bool(close_px > ref["prior_ath"])
        out["close_below_prior_ath"] = bool(close_px < ref["prior_ath"])
    return out


def _empty_outcomes() -> pd.DataFrame:
    """Empty result that still carries the join key.

    A bare ``pd.DataFrame([])`` has no columns, which makes callers' merge on
    ``session_date`` raise KeyError instead of yielding "no matches".
    """
    return pd.DataFrame({"session_date": pd.Series([], dtype="datetime64[ns]"),
                         "cutoff_min": pd.Series([], dtype="int64")})


def build_outcomes(master: pd.DataFrame, daily: pd.DataFrame, cutoff_min: int,
                   sessions: list, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    if len(sessions) == 0 or len(master) == 0 or len(daily) == 0:
        return _empty_outcomes()
    ms = sb.attach_sessions(master, cfg)
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0)]
    groups = dict(tuple(cash.groupby("session_date")))
    daily_idx = daily.set_index("session_date")
    rows = []
    for sd in sessions:
        if sd not in groups or sd not in daily_idx.index:
            continue
        r = daily_idx.loc[sd]
        ref = {"cash_open": r["cash_open_adj"], "pdh": r["previous_day_high"],
               "pdl": r["previous_day_low"], "prev_close": r["previous_close"],
               "overnight_high": r["overnight_high_adj"], "overnight_low": r["overnight_low_adj"],
               "prior_ath": r["prior_ath"]}
        o = session_outcome(groups[sd], ref, cutoff_min, cfg)
        o["session_date"] = sd
        rows.append(o)
    return pd.DataFrame(rows) if rows else _empty_outcomes()
