"""Categorical market-context and opening-pattern recognition.

The ATH filter was a *threshold* ("within 0.25% of the high").  That is one value
of a more general question -- **where are we, and how did we get here** -- so
this module replaces thresholds with named categories a trader would actually
say out loud:

    location      AT_ATH · NEAR_ATH · BREAKOUT_UP · UPPER_RANGE · MID_RANGE ·
                  LOWER_RANGE · BREAKDOWN · AT_LOWS
    swing         STRONG_BULL · BULL · RANGE · BEAR · STRONG_BEAR
    prior_day     TREND_UP · TREND_DOWN · REVERSAL_UP · REVERSAL_DOWN ·
                  BIG_RANGE (news/shock) · RANGE · INSIDE · OUTSIDE
    gap           LARGE_UP · MODERATE_UP · SMALL_UP · FLAT · SMALL_DOWN ·
                  MODERATE_DOWN · LARGE_DOWN
    open_loc      ABOVE_PDH · UPPER_PD · MID_PD · LOWER_PD · BELOW_PDL
    overnight     TREND_UP · TREND_DOWN · RANGE · REVERSAL_UP · REVERSAL_DOWN
    opening       DRIVE_UP · DRIVE_DOWN · REJECTION_UP · REJECTION_DOWN ·
                  RANGE_OPEN
    ma_state      ABOVE_RISING · ABOVE_FALLING · BELOW_RISING · BELOW_FALLING

Inputs are restricted to **OHLC and moving averages**.  No oscillators, no
volume, no ATR-style normalisation inside the opening-pattern tests -- those use
the bars' own ranges.  Where a "big" or "small" judgement is needed the cut is a
**rolling percentile of the instrument's own recent history**, computed from
prior sessions only, so the same label means the same thing in 2003 and 2026.

Every classifier is prior-only except ``opening`` which is clamped at the
observation cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb


# --------------------------------------------------------------------------- #
# category vocabularies
# --------------------------------------------------------------------------- #
LOCATION = ["AT_ATH", "NEAR_ATH", "BREAKOUT_UP", "UPPER_RANGE", "MID_RANGE",
            "LOWER_RANGE", "BREAKDOWN", "AT_LOWS"]
SWING = ["STRONG_BULL", "BULL", "RANGE", "BEAR", "STRONG_BEAR"]
PRIOR_DAY = ["TREND_UP", "TREND_DOWN", "REVERSAL_UP", "REVERSAL_DOWN",
             "BIG_RANGE", "RANGE", "INSIDE", "OUTSIDE"]
GAP = ["LARGE_UP", "MODERATE_UP", "SMALL_UP", "FLAT",
       "SMALL_DOWN", "MODERATE_DOWN", "LARGE_DOWN"]
OPEN_LOC = ["ABOVE_PDH", "UPPER_PD", "MID_PD", "LOWER_PD", "BELOW_PDL"]
OVERNIGHT = ["TREND_UP", "TREND_DOWN", "RANGE", "REVERSAL_UP", "REVERSAL_DOWN"]
OPENING = ["DRIVE_UP", "DRIVE_DOWN", "REJECTION_UP", "REJECTION_DOWN", "RANGE_OPEN"]
MA_STATE = ["ABOVE_RISING", "ABOVE_FALLING", "BELOW_RISING", "BELOW_FALLING"]


@dataclass
class ContextParams:
    """Thresholds behind the labels.  Editable, but they are *definitions* of the
    categories rather than knobs to tune per query."""
    # location
    ath_at_pct: float = 0.25          # within this of the prior ATH -> AT_ATH
    ath_near_pct: float = 1.00        # ... -> NEAR_ATH
    range_lookback: int = 20          # sessions for the "recent range"
    long_lookback: int = 250          # sessions for the 52-week style extremes
    # swing / moving averages (daily closes)
    ma_fast: int = 20
    ma_slow: int = 50
    slope_window: int = 5             # MA slope measured over this many sessions
    slope_flat_frac: float = 0.15     # |slope| below this * recent daily range -> flat
    # prior-day character
    big_range_pctl: float = 90.0      # prior-day range above this pctl -> BIG_RANGE (news/shock)
    trend_close_loc: float = 0.70     # close in top/bottom of its range -> trend day
    quiet_range_pctl: float = 40.0
    # gap buckets, as percentiles of the |gap| distribution of prior sessions
    gap_flat_pctl: float = 20.0
    gap_small_pctl: float = 50.0
    gap_moderate_pctl: float = 80.0
    # open location inside the prior-day range
    pd_upper_frac: float = 0.67
    pd_lower_frac: float = 0.33
    # overnight character
    on_trend_close_loc: float = 0.70
    # opening pattern (first N minutes, pure bar geometry)
    opening_window_min: int = 15
    drive_close_loc: float = 0.67     # closed in top third of the window range
    reject_close_loc: float = 0.33    # ... or bottom third
    # A rejection means price travelled MEANINGFULLY one way before closing back
    # the other.  Without this a single tick past the open would count, and then
    # almost every session is a "rejection".  Measured against the window's own
    # range, so it stays pure geometry.
    excursion_frac: float = 0.35
    # rolling window used for every percentile cut
    pctl_window: int = 250

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PARAMS = ContextParams()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _prior_pctl(s: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling percentile of PRIOR values only (current value excluded)."""
    return s.shift(1).rolling(window, min_periods=max(20, window // 5)).quantile(q / 100.0)


def _safe(v, default=np.nan):
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


# --------------------------------------------------------------------------- #
# 1. daily context: location, swing regime, moving-average state
# --------------------------------------------------------------------------- #
def daily_context(daily: pd.DataFrame, p: ContextParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Per-session categorical context from daily OHLC + moving averages.

    Everything is shifted so a session is described by what was knowable *before*
    it opened.
    """
    d = daily.sort_values("session_date").reset_index(drop=True).copy()
    close = d["cash_close_adj"]
    high, low = d["cash_high_adj"], d["cash_low_adj"]

    # ---- moving averages on daily closes, prior-only ----
    ma_f = close.shift(1).rolling(p.ma_fast, min_periods=p.ma_fast // 2).mean()
    ma_s = close.shift(1).rolling(p.ma_slow, min_periods=p.ma_slow // 2).mean()
    d["ma_fast"], d["ma_slow"] = ma_f, ma_s
    slope = ma_f - ma_f.shift(p.slope_window)
    daily_rng = (high - low).shift(1).rolling(p.ma_fast, min_periods=5).mean()
    d["ma_fast_slope"] = slope
    flat = slope.abs() < (p.slope_flat_frac * daily_rng)

    # the reference price for "where are we" is the session's OPEN (knowable at 09:00)
    ref = d["cash_open_adj"]

    # ---- moving-average state ----
    above = ref > ma_f
    rising = slope > 0
    d["ma_state"] = np.where(above & rising, "ABOVE_RISING",
                     np.where(above & ~rising, "ABOVE_FALLING",
                      np.where(~above & rising, "BELOW_RISING", "BELOW_FALLING")))
    d.loc[ma_f.isna(), "ma_state"] = "UNKNOWN"

    # ---- recent range, prior-only ----
    hi_n = high.shift(1).rolling(p.range_lookback, min_periods=5).max()
    lo_n = low.shift(1).rolling(p.range_lookback, min_periods=5).min()
    hi_l = high.shift(1).rolling(p.long_lookback, min_periods=40).max()
    lo_l = low.shift(1).rolling(p.long_lookback, min_periods=40).min()
    d["range_high"], d["range_low"] = hi_n, lo_n
    span = (hi_n - lo_n).replace(0, np.nan)
    d["location_in_range"] = (ref - lo_n) / span

    # ---- location, with the ATH categories first ----
    ath = d["prior_ath"]
    dist_ath = 100.0 * (ref - ath) / ath
    d["distance_to_ath_pct"] = dist_ath
    dist_low = 100.0 * (ref - lo_l) / lo_l

    loc = pd.Series("MID_RANGE", index=d.index, dtype=object)
    loc[d["location_in_range"] <= p.pd_lower_frac] = "LOWER_RANGE"
    loc[d["location_in_range"] >= p.pd_upper_frac] = "UPPER_RANGE"
    loc[ref > hi_n] = "BREAKOUT_UP"
    loc[ref < lo_n] = "BREAKDOWN"
    loc[dist_low.abs() <= p.ath_near_pct] = "AT_LOWS"
    loc[dist_ath.abs() <= p.ath_near_pct] = "NEAR_ATH"
    loc[dist_ath >= -p.ath_at_pct] = "AT_ATH"          # at, or above, the old high
    loc[ath.isna()] = "UNKNOWN"
    d["location"] = loc

    # ---- swing regime ----
    stacked_up = (ma_f > ma_s) & (ref > ma_f)
    stacked_dn = (ma_f < ma_s) & (ref < ma_f)
    swing = pd.Series("RANGE", index=d.index, dtype=object)
    swing[stacked_up & rising] = "BULL"
    swing[stacked_dn & ~rising] = "BEAR"
    swing[stacked_up & rising & (ref > hi_n)] = "STRONG_BULL"
    swing[stacked_dn & ~rising & (ref < lo_n)] = "STRONG_BEAR"
    swing[flat] = "RANGE"
    swing[ma_s.isna()] = "UNKNOWN"
    d["swing"] = swing
    return d


# --------------------------------------------------------------------------- #
# 2. prior-day character (including the news/shock footprint)
# --------------------------------------------------------------------------- #
def prior_day_context(daily: pd.DataFrame, p: ContextParams = DEFAULT_PARAMS) -> pd.DataFrame:
    d = daily.sort_values("session_date").reset_index(drop=True).copy()
    o, h, l, c = (d["cash_open_adj"], d["cash_high_adj"],
                  d["cash_low_adj"], d["cash_close_adj"])
    rng = (h - l)
    close_loc = ((c - l) / rng.replace(0, np.nan))

    big = _prior_pctl(rng, p.pctl_window, p.big_range_pctl)
    quiet = _prior_pctl(rng, p.pctl_window, p.quiet_range_pctl)

    inside = (h < h.shift(1)) & (l > l.shift(1))
    outside = (h > h.shift(1)) & (l < l.shift(1))
    # a reversal day trades beyond the prior extreme and closes back the other way
    rev_up = (l < l.shift(1)) & (close_loc >= p.trend_close_loc)
    rev_dn = (h > h.shift(1)) & (close_loc <= 1 - p.trend_close_loc)

    t = pd.Series("RANGE", index=d.index, dtype=object)
    t[close_loc >= p.trend_close_loc] = "TREND_UP"
    t[close_loc <= 1 - p.trend_close_loc] = "TREND_DOWN"
    t[rng <= quiet] = "RANGE"
    t[inside] = "INSIDE"
    t[outside] = "OUTSIDE"
    t[rev_up] = "REVERSAL_UP"
    t[rev_dn] = "REVERSAL_DOWN"
    # an outsized range dominates every other label -- that is the news footprint
    t[rng >= big] = "BIG_RANGE"

    out = pd.DataFrame({"session_date": d["session_date"]})
    out["day_type"] = t
    out["day_range"] = rng
    out["day_close_loc"] = close_loc
    # shift so each row carries YESTERDAY's character
    out["prior_day_type"] = out["day_type"].shift(1)
    out["prior_day_close_loc"] = close_loc.shift(1)
    out["prior_day_was_big_range"] = (rng >= big).shift(1).fillna(False)
    # how many of the last 3 sessions were range-ish -> "range for the past few days"
    isr = out["day_type"].isin(["RANGE", "INSIDE"]).astype(float)
    out["prior_3d_range_count"] = isr.shift(1).rolling(3, min_periods=1).sum()
    return out


# --------------------------------------------------------------------------- #
# 3. gap + open location + overnight character
# --------------------------------------------------------------------------- #
def open_context(daily: pd.DataFrame, p: ContextParams = DEFAULT_PARAMS) -> pd.DataFrame:
    d = daily.sort_values("session_date").reset_index(drop=True).copy()
    gap = d["gap_pct"]
    a = gap.abs()
    q_flat = _prior_pctl(a, p.pctl_window, p.gap_flat_pctl)
    q_small = _prior_pctl(a, p.pctl_window, p.gap_small_pctl)
    q_mod = _prior_pctl(a, p.pctl_window, p.gap_moderate_pctl)

    g = pd.Series("FLAT", index=d.index, dtype=object)
    up, dn = gap > 0, gap < 0
    g[up & (a > q_flat)] = "SMALL_UP"
    g[up & (a > q_small)] = "MODERATE_UP"
    g[up & (a > q_mod)] = "LARGE_UP"
    g[dn & (a > q_flat)] = "SMALL_DOWN"
    g[dn & (a > q_small)] = "MODERATE_DOWN"
    g[dn & (a > q_mod)] = "LARGE_DOWN"
    g[q_flat.isna()] = "UNKNOWN"

    # open location relative to the prior day's range
    pdh, pdl = d["previous_day_high"], d["previous_day_low"]
    frac = (d["cash_open_adj"] - pdl) / (pdh - pdl).replace(0, np.nan)
    ol = pd.Series("MID_PD", index=d.index, dtype=object)
    ol[frac <= p.pd_lower_frac] = "LOWER_PD"
    ol[frac >= p.pd_upper_frac] = "UPPER_PD"
    ol[d["cash_open_adj"] > pdh] = "ABOVE_PDH"
    ol[d["cash_open_adj"] < pdl] = "BELOW_PDL"
    ol[pdh.isna()] = "UNKNOWN"

    # overnight character from its own OHLC
    oo, oh, ol_, oc = (d["overnight_open_adj"], d["overnight_high_adj"],
                       d["overnight_low_adj"], d["overnight_close_adj"])
    on_rng = (oh - ol_).replace(0, np.nan)
    on_loc = (oc - ol_) / on_rng
    # give-back each way, as a fraction of the overnight range.  A reversal has
    # to have travelled meaningfully against its eventual direction first --
    # otherwise every trending overnight is mislabelled a reversal.
    on_dn_ext = (oo - ol_) / on_rng
    on_up_ext = (oh - oo) / on_rng
    trend_up = on_loc >= p.on_trend_close_loc
    trend_dn = on_loc <= 1 - p.on_trend_close_loc
    on = pd.Series("RANGE", index=d.index, dtype=object)
    on[trend_up] = "TREND_UP"
    on[trend_dn] = "TREND_DOWN"
    on[trend_up & (on_dn_ext >= p.excursion_frac)] = "REVERSAL_UP"
    on[trend_dn & (on_up_ext >= p.excursion_frac)] = "REVERSAL_DOWN"
    on[on_rng.isna()] = "UNKNOWN"

    return pd.DataFrame({
        "session_date": d["session_date"],
        "gap_bucket": g, "open_loc": ol, "overnight_type": on,
        "open_frac_in_pd_range": frac,
        "overnight_close_loc": on_loc,
    })


# --------------------------------------------------------------------------- #
# 4. opening pattern (cutoff-clamped, pure bar geometry)
# --------------------------------------------------------------------------- #
def classify_opening(bars: pd.DataFrame, cash_open: float,
                     p: ContextParams = DEFAULT_PARAMS,
                     ema: float | None = None) -> dict:
    """Name the shape of the first ``opening_window_min`` minutes.

    DRIVE_*      went one way and closed there, without giving it back
    REJECTION_*  poked one way, closed back through the open the other way
    RANGE_OPEN   two-sided, closed mid
    """
    w = bars[(bars["mso"] >= 0) & (bars["mso"] < p.opening_window_min)].sort_values("mso")
    if len(w) == 0:
        return {}
    O = w["adj_open"].to_numpy(); H = w["adj_high"].to_numpy()
    L = w["adj_low"].to_numpy();  C = w["adj_close"].to_numpy()
    hi, lo, c = float(H.max()), float(L.min()), float(C[-1])
    rng = hi - lo
    loc = (c - lo) / rng if rng > 0 else np.nan

    # excursion each way, as a fraction of the window's own range
    up_ext = (hi - cash_open) / rng if rng > 0 else np.nan
    dn_ext = (cash_open - lo) / rng if rng > 0 else np.nan
    above = c > cash_open
    below = c < cash_open
    big_up = up_ext >= p.excursion_frac
    big_dn = dn_ext >= p.excursion_frac

    if below and loc <= p.reject_close_loc and big_up:
        label = "REJECTION_UP"        # bull trap: ran up, sold back through the open
    elif above and loc >= p.drive_close_loc and big_dn:
        label = "REJECTION_DOWN"      # bear trap
    elif above and loc >= p.drive_close_loc:
        label = "DRIVE_UP"            # went up and stayed, no real give-back
    elif below and loc <= p.reject_close_loc:
        label = "DRIVE_DOWN"
    else:
        label = "RANGE_OPEN"

    out = {
        "opening": label,
        "op_window_high": hi, "op_window_low": lo, "op_window_close": c,
        "op_close_loc": loc,
        "op_up_excursion": up_ext, "op_down_excursion": dn_ext,
        "op_poked_above_open": bool(hi > cash_open),
        "op_poked_below_open": bool(lo < cash_open),
        "op_close_above_open": bool(above),
        "op_n_bear_bars": int((C < O).sum()),
        "op_n_bull_bars": int((C > O).sum()),
        "op_lower_highs": bool(len(H) >= 3 and np.all(np.diff(H) < 0)),
        "op_higher_lows": bool(len(L) >= 3 and np.all(np.diff(L) > 0)),
        "op_close_below_prior_low": bool(len(C) >= 2 and np.any(C[1:] < L[:-1])),
        "op_close_above_prior_high": bool(len(C) >= 2 and np.any(C[1:] > H[:-1])),
    }
    if ema is not None and not pd.isna(ema):
        out["op_close_vs_ema"] = "ABOVE" if c > ema else "BELOW"
    return out


def intraday_ema(bars: pd.DataFrame, span: int, upto_mso: int) -> float:
    """EMA of 5m closes up to (and including) ``upto_mso`` -- never beyond."""
    w = bars[(bars["mso"] >= 0) & (bars["mso"] <= upto_mso)].sort_values("mso")
    if len(w) == 0:
        return np.nan
    return float(w["adj_close"].ewm(span=span, adjust=False).mean().iloc[-1])


# --------------------------------------------------------------------------- #
# 5. assemble one table
# --------------------------------------------------------------------------- #
def build_context_table(master: pd.DataFrame, daily: pd.DataFrame,
                        cfg: Config = DEFAULT_CONFIG,
                        p: ContextParams = DEFAULT_PARAMS,
                        decision_min: int = 30,
                        ema_span: int = 20) -> pd.DataFrame:
    """One row per session with every category attached."""
    dc = daily_context(daily, p)
    pdc = prior_day_context(daily, p)
    oc = open_context(daily, p)
    base = (dc[["session_date", "location", "swing", "ma_state", "ma_fast", "ma_slow",
                "ma_fast_slope", "location_in_range", "distance_to_ath_pct",
                "range_high", "range_low"]]
            .merge(pdc[["session_date", "prior_day_type", "prior_day_close_loc",
                        "prior_day_was_big_range", "prior_3d_range_count"]], on="session_date")
            .merge(oc, on="session_date"))

    ms = sb.attach_sessions(master, cfg)
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0)]
    groups = dict(tuple(cash.groupby("session_date")))
    di = daily.set_index("session_date")

    rows = []
    for sd, g in groups.items():
        if sd not in di.index:
            continue
        o = di.loc[sd, "cash_open_adj"]
        if pd.isna(o):
            continue
        g = g.sort_values("mso")
        if len(g[g["mso"] < decision_min]) < 2:
            continue
        ema = intraday_ema(g, ema_span, p.opening_window_min - 1)
        rec = {"session_date": sd}
        rec.update(classify_opening(g, o, p, ema))
        rec["op_ema"] = ema
        rows.append(rec)
    op = pd.DataFrame(rows)
    out = base.merge(op, on="session_date", how="inner")
    return out.sort_values("session_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. category-based filter
# --------------------------------------------------------------------------- #
@dataclass
class ContextSpec:
    """Pick categories, not thresholds.  ``None`` means "don't care"."""
    location: tuple[str, ...] | None = None
    swing: tuple[str, ...] | None = None
    ma_state: tuple[str, ...] | None = None
    prior_day_type: tuple[str, ...] | None = None
    gap_bucket: tuple[str, ...] | None = None
    open_loc: tuple[str, ...] | None = None
    overnight_type: tuple[str, ...] | None = None
    opening: tuple[str, ...] | None = None
    # optional extra structure, still pure geometry
    require_lower_highs: bool = False
    require_higher_lows: bool = False
    require_close_beyond_prior_bar: bool = False
    min_prior_3d_range_count: float | None = None

    FIELDS = ("location", "swing", "ma_state", "prior_day_type", "gap_bucket",
              "open_loc", "overnight_type", "opening")

    def describe(self) -> list[str]:
        d = []
        for f in self.FIELDS:
            v = getattr(self, f)
            if v:
                d.append(f"{f} in {list(v)}")
        if self.require_lower_highs:
            d.append("lower highs in the opening window")
        if self.require_higher_lows:
            d.append("higher lows in the opening window")
        if self.require_close_beyond_prior_bar:
            d.append("a bar closed beyond the prior bar's extreme")
        if self.min_prior_3d_range_count is not None:
            d.append(f"≥{self.min_prior_3d_range_count:g} of the last 3 days were range days")
        return d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_context(table: pd.DataFrame, spec: ContextSpec) -> pd.Series:
    mask = pd.Series(True, index=table.index)
    for f in ContextSpec.FIELDS:
        v = getattr(spec, f)
        if v and f in table:
            mask &= table[f].isin(v)
    if spec.require_lower_highs and "op_lower_highs" in table:
        mask &= table["op_lower_highs"].fillna(False)
    if spec.require_higher_lows and "op_higher_lows" in table:
        mask &= table["op_higher_lows"].fillna(False)
    if spec.require_close_beyond_prior_bar:
        col = ("op_close_below_prior_low"
               if (spec.opening and "REJECTION_UP" in spec.opening or
                   spec.opening and "DRIVE_DOWN" in spec.opening)
               else "op_close_above_prior_high")
        if col in table:
            mask &= table[col].fillna(False)
    if spec.min_prior_3d_range_count is not None and "prior_3d_range_count" in table:
        mask &= table["prior_3d_range_count"] >= spec.min_prior_3d_range_count
    return mask.fillna(False)


def context_funnel(table: pd.DataFrame, spec: ContextSpec) -> pd.DataFrame:
    rows = [{"step": "all sessions", "surviving": len(table), "cut": 0}]
    mask = pd.Series(True, index=table.index)
    prev = len(table)
    for f in ContextSpec.FIELDS:
        v = getattr(spec, f)
        if not v or f not in table:
            continue
        mask &= table[f].isin(v)
        n = int(mask.sum())
        rows.append({"step": f"{f} in {list(v)}", "surviving": n, "cut": prev - n})
        prev = n
    for flag, col, label in (
        ("require_lower_highs", "op_lower_highs", "lower highs"),
        ("require_higher_lows", "op_higher_lows", "higher lows"),
    ):
        if getattr(spec, flag) and col in table:
            mask &= table[col].fillna(False)
            n = int(mask.sum())
            rows.append({"step": label, "surviving": n, "cut": prev - n})
            prev = n
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the ATH bull-trap, expressed in the new vocabulary
# --------------------------------------------------------------------------- #
ATH_TRAP_CONTEXT = ContextSpec(
    location=("AT_ATH", "NEAR_ATH"),
    swing=("BULL", "STRONG_BULL"),
    gap_bucket=("MODERATE_UP", "LARGE_UP"),
    open_loc=("ABOVE_PDH",),
    overnight_type=("TREND_UP",),
    opening=("REJECTION_UP",),
)
