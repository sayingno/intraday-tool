"""All-time-high engine (futures, roll-adjusted).

Everything is computed on the continuous BACK-ADJUSTED price (``adj_*``) so no raw
levels are ever mixed across contracts.  The single hard rule: ``prior_ath[D]``
uses only sessions strictly before D -- no bar from session D or later ever
enters it.  This is the project's main anti-leakage guarantee and is unit-tested.

Definitions (from the spec):
    prior_ath[D]                     = max adjusted intraday high over sessions < D
    ath_at_open                      = cash_open_adj > prior_ath[D]
    distance_open_to_ath_pct         = (cash_open_adj - prior_ath) / prior_ath
    distance_overnight_high_to_ath   = (overnight_high_adj - prior_ath) / prior_ath
    ath_reached_overnight            = overnight_high_adj >= prior_ath[D]
    ath_broken_first_{5,15,30}m      = any cash bar in the window with adj_high > prior_ath
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb


def compute_prior_ath(daily_ohlc: pd.DataFrame) -> pd.Series:
    """prior_ath[D] = expanding max of the adjusted day high over sessions < D.

    Uses ``.shift(1)`` after the running max so session D's own high can never
    enter its own prior-ATH -- the core no-future-leakage guarantee.
    """
    d = daily_ohlc.sort_values("session_date")
    running = d["day_high_adj"].cummax()
    prior = running.shift(1)
    return pd.Series(prior.values, index=d["session_date"].values, name="prior_ath")


def _first_window_breaks(master: pd.DataFrame, prior_ath: pd.Series,
                         cfg: Config, windows=(5, 15, 30)) -> pd.DataFrame:
    """Whether the adjusted high exceeds prior_ath within the first N minutes."""
    ms = sb.attach_sessions(master, cfg)
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0)].copy()
    cash["prior_ath"] = cash["session_date"].map(prior_ath)
    cash["broke"] = cash["adj_high"] > cash["prior_ath"]
    out = {}
    for n in windows:
        w = cash[cash["mso"] < n]
        out[f"ath_broken_first_{n}m"] = w.groupby("session_date")["broke"].any()
    res = pd.DataFrame(out)
    return res


def compute_ath_table(master: pd.DataFrame, daily_ohlc: pd.DataFrame,
                      cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Per-session ATH feature table (all on adjusted prices)."""
    d = daily_ohlc.sort_values("session_date").reset_index(drop=True).copy()
    prior = compute_prior_ath(d)
    d["prior_ath"] = d["session_date"].map(prior)
    d["all_time_high_incl"] = d["day_high_adj"].cummax()

    co = d["cash_open_adj"]
    d["ath_at_open"] = co > d["prior_ath"]
    d["distance_open_to_ath_pct"] = 100.0 * (co - d["prior_ath"]) / d["prior_ath"]
    d["distance_overnight_high_to_ath_pct"] = (
        100.0 * (d["overnight_high_adj"] - d["prior_ath"]) / d["prior_ath"])
    d["ath_reached_overnight"] = d["overnight_high_adj"] >= d["prior_ath"]

    # near-ATH within each configured tolerance (absolute distance at the open)
    for tol in cfg.ath_tolerances_pct:
        d[f"near_ath_{tol:g}pct"] = d["distance_open_to_ath_pct"].abs() <= tol

    # first-window ATH breaks
    breaks = _first_window_breaks(master, prior, cfg)
    d = d.merge(breaks, left_on="session_date", right_index=True, how="left")
    for c in ("ath_broken_first_5m", "ath_broken_first_15m", "ath_broken_first_30m"):
        if c not in d:
            d[c] = False
        d[c] = d[c].fillna(False).astype(bool)

    return d


def ath_columns() -> list[str]:
    return [
        "prior_ath", "all_time_high_incl", "ath_at_open",
        "distance_open_to_ath_pct", "distance_overnight_high_to_ath_pct",
        "ath_reached_overnight", "ath_broken_first_5m", "ath_broken_first_15m",
        "ath_broken_first_30m",
    ]
