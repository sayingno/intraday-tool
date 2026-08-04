"""Opening-pattern / price-action features, all derived from OHLC only.

Everything is measured on the back-adjusted price and STRICTLY within the
observation window [cash_open, cutoff].  No bar at ``mso > cutoff`` is ever read
here -- that is the anti-leakage contract for the matching stage.

Feature families
----------------
* per-window structure   (5 / 15 / 30 / 60 min + the chosen cutoff)
* first-5m-bar features
* post-spike follow-through   (the crux of the "weak follow-through" pattern)
* bearish-weakness events     (objective, OHLC-derived, with timestamps)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb


# --------------------------------------------------------------------------- #
# small numeric helpers (operate on numpy arrays of a single window)
# --------------------------------------------------------------------------- #
def _safe_div(a, b):
    return float(a) / float(b) if b not in (0, 0.0, None) and not pd.isna(b) else np.nan


def directional_efficiency(closes: np.ndarray, open_px: float) -> float:
    """|net move| / sum|bar-to-bar move|.  Smooth trend -> ~1, chop -> ~0."""
    path = np.concatenate([[open_px], closes])
    steps = np.abs(np.diff(path))
    denom = steps.sum()
    if denom == 0:
        return np.nan
    return abs(path[-1] - path[0]) / denom


def overlap_ratio(highs: np.ndarray, lows: np.ndarray) -> float:
    """Mean overlap of consecutive bars / their max range.  High => congestion."""
    if len(highs) < 2:
        return np.nan
    ov = []
    for i in range(1, len(highs)):
        inter = min(highs[i], highs[i - 1]) - max(lows[i], lows[i - 1])
        rng = max(highs[i] - lows[i], highs[i - 1] - lows[i - 1])
        if rng > 0:
            ov.append(max(0.0, inter) / rng)
    return float(np.mean(ov)) if ov else np.nan


def _swings(highs, lows, k):
    """Indices of swing highs / lows (local extremum over +-k bars)."""
    sh, sl = [], []
    n = len(highs)
    for i in range(n):
        lo = max(0, i - k); hi = min(n, i + k + 1)
        if highs[i] == np.max(highs[lo:hi]) and (hi - lo) > 1:
            sh.append(i)
        if lows[i] == np.min(lows[lo:hi]) and (hi - lo) > 1:
            sl.append(i)
    return sh, sl


# --------------------------------------------------------------------------- #
# window structure
# --------------------------------------------------------------------------- #
def window_structure(o, h, l, c, mso, ref: dict) -> dict:
    if len(o) == 0:
        return {}
    open_px = ref["cash_open"]
    hi, lo = float(np.max(h)), float(np.min(l))
    rng = hi - lo
    net = float(c[-1]) - open_px
    bull = int(np.sum(c > o)); bear = int(np.sum(c < o))
    # running peak/trough for pullback/bounce
    run_max = np.maximum.accumulate(h)
    run_min = np.minimum.accumulate(l)
    max_pullback = float(np.max(run_max - l))          # deepest drop from a prior peak
    max_bounce = float(np.max(h - run_min))            # biggest pop from a prior trough
    up_move = hi - open_px
    down_move = open_px - lo
    hh = int(np.sum(np.diff(h) > 0)); lh = int(np.sum(np.diff(h) < 0))
    hl = int(np.sum(np.diff(l) > 0)); ll = int(np.sum(np.diff(l) < 0))
    return {
        "net_return_pct": 100.0 * net / open_px,
        "net_points": net,
        "high_low_range": rng,
        "range_normalized_by_prior_daily_range": _safe_div(rng, ref.get("prev_day_range")),
        "range_normalized_by_ATR": _safe_div(rng, ref.get("atr")),
        "number_of_bull_bars": bull,
        "number_of_bear_bars": bear,
        "percentage_bull_bars": 100.0 * bull / len(o),
        "number_of_higher_highs": hh, "number_of_lower_highs": lh,
        "number_of_higher_lows": hl, "number_of_lower_lows": ll,
        "close_location_in_window_range": _safe_div(c[-1] - lo, rng),
        "maximum_pullback": max_pullback,
        "maximum_pullback_as_fraction_of_up_move": _safe_div(max_pullback, up_move),
        "maximum_bounce_as_fraction_of_down_move": _safe_div(max_bounce, down_move),
        "directional_efficiency": directional_efficiency(c, open_px),
        "bar_overlap_ratio": overlap_ratio(h, l),
        "time_above_cash_open": float(np.mean(c > open_px)),
        "time_below_cash_open": float(np.mean(c < open_px)),
        "distance_from_cash_open": net,
        "distance_from_PDH": float(c[-1] - ref.get("pdh", np.nan)),
        "distance_from_PDL": float(c[-1] - ref.get("pdl", np.nan)),
        "distance_from_previous_close": float(c[-1] - ref.get("prev_close", np.nan)),
        "distance_from_overnight_high": float(c[-1] - ref.get("overnight_high", np.nan)),
        "distance_from_overnight_low": float(c[-1] - ref.get("overnight_low", np.nan)),
    }


def first_bar_features(o, h, l, c, ref: dict) -> dict:
    if len(o) == 0:
        return {}
    o0, h0, l0, c0 = float(o[0]), float(h[0]), float(l[0]), float(c[0])
    rng = h0 - l0
    body = c0 - o0
    return {
        "first_bar_return": 100.0 * (c0 - o0) / o0,
        "first_bar_range": rng,
        "first_bar_range_normalized": _safe_div(rng, ref.get("atr")),
        "first_bar_range_norm_prevrange": _safe_div(rng, ref.get("prev_day_range")),
        "first_bar_body": body,
        "first_bar_body_to_range": _safe_div(abs(body), rng),
        "first_bar_upper_wick": h0 - max(o0, c0),
        "first_bar_lower_wick": min(o0, c0) - l0,
        "first_bar_close_location": _safe_div(c0 - l0, rng),
        "first_bar_breaks_PDH": bool(h0 > ref.get("pdh", np.inf)),
        "first_bar_breaks_overnight_high": bool(h0 > ref.get("overnight_high", np.inf)),
        "first_bar_breaks_prior_ATH": bool(h0 > ref.get("prior_ath", np.inf)),
        "first_bar_is_bull": bool(c0 > o0),
    }


def post_spike_features(o, h, l, c, mso, ref: dict, cfg: Config) -> dict:
    """Follow-through AFTER the first bar (the 'weak follow-through' signature)."""
    if len(o) < 2:
        return {"extension_ratio": np.nan, "post_spike_pullback_ratio": np.nan,
                "post_spike_overlap_ratio": np.nan, "post_spike_efficiency": np.nan,
                "number_of_new_highs_after_first_bar": 0,
                "average_distance_between_new_highs": np.nan,
                "minutes_until_next_meaningful_new_high": np.nan,
                "close_progress_ratio": np.nan}
    fb_high = float(h[0]); fb_low = float(l[0]); fb_close = float(c[0])
    fb_range = fb_high - fb_low
    post_h, post_l, post_c = h[1:], l[1:], c[1:]
    post_mso = mso[1:]
    add_ext = float(np.max(post_h)) - fb_high
    pull = fb_high - float(np.min(post_l))
    # new highs after bar1 (relative to running max seeded at fb_high)
    run = fb_high
    new_high_times = []
    meaningful = cfg.meaningful_move_atr_frac * (ref.get("atr") or np.nan)
    first_meaningful = np.nan
    for hh, mm in zip(post_h, post_mso):
        if hh > run:
            new_high_times.append(mm)
            if np.isnan(first_meaningful) and (hh - fb_high) >= (meaningful if not pd.isna(meaningful) else 0):
                first_meaningful = mm
            run = hh
    if len(new_high_times) >= 2:
        avg_gap = float(np.mean(np.diff(new_high_times)))
    else:
        avg_gap = np.nan
    return {
        "additional_extension_after_first_bar": add_ext,
        "extension_ratio": _safe_div(add_ext, fb_range),
        "post_spike_pullback": pull,
        "post_spike_pullback_ratio": _safe_div(pull, fb_range),
        "post_spike_overlap_ratio": overlap_ratio(post_h, post_l),
        "post_spike_efficiency": directional_efficiency(post_c, fb_close),
        "number_of_new_highs_after_first_bar": len(new_high_times),
        "average_distance_between_new_highs": avg_gap,
        "minutes_until_next_meaningful_new_high": (first_meaningful - 0) if not pd.isna(first_meaningful) else np.nan,
        "close_progress_ratio": _safe_div(float(c[-1]) - fb_close, fb_range),
    }


def weakness_features(o, h, l, c, mso, ref: dict, cfg: Config) -> dict:
    """Objective bearish-weakness events with timestamps (minutes since open)."""
    n = len(o)
    out = {
        "first_close_below_prev2_low_min": np.nan,
        "first_lower_swing_high_min": np.nan,
        "first_lower_swing_low_min": np.nan,
        "first_bearish_expansion_min": np.nan,
        "first_bullish_expansion_min": np.nan,
        "first_break_first15m_low_min": np.nan,
        "first_break_first30m_low_min": np.nan,
        "first_stall_break_min": np.nan,
        "first_meaningful_lower_high_min": np.nan,
        "first_meaningful_higher_low_min": np.nan,
        "failed_to_reclaim_bear_expansion_high": False,
        "time_below_open_frac": float(np.mean(c < ref["cash_open"])) if n else np.nan,
        "opening_range_break_direction": 0,
        "first_opening_range_break_min": np.nan,
    }
    if n == 0:
        return out
    ranges = h - l
    med_range = np.median(ranges[:min(n, cfg.expansion_median_window)]) if n else np.nan
    exp_thresh = cfg.expansion_bar_multiple * med_range

    # bearish / bullish expansion bars
    for i in range(n):
        if pd.isna(out["first_bearish_expansion_min"]) and c[i] < o[i] and ranges[i] > exp_thresh:
            out["first_bearish_expansion_min"] = mso[i]
            # failure to reclaim its high afterwards
            if i + 1 < n:
                out["failed_to_reclaim_bear_expansion_high"] = bool(np.all(c[i + 1:] < h[i]))
            else:
                out["failed_to_reclaim_bear_expansion_high"] = True
        if pd.isna(out["first_bullish_expansion_min"]) and c[i] > o[i] and ranges[i] > exp_thresh:
            out["first_bullish_expansion_min"] = mso[i]

    # first close below the low of the previous two bars
    for i in range(2, n):
        if c[i] < min(l[i - 1], l[i - 2]):
            out["first_close_below_prev2_low_min"] = mso[i]; break

    # opening-range (first bar) break
    fb_high, fb_low = h[0], l[0]
    for i in range(1, n):
        if h[i] > fb_high:
            out["opening_range_break_direction"] = 1
            out["first_opening_range_break_min"] = mso[i]; break
        if l[i] < fb_low:
            out["opening_range_break_direction"] = -1
            out["first_opening_range_break_min"] = mso[i]; break

    # breaks below first-15m / first-30m lows
    def low_by(minute):
        m = mso < minute
        return np.min(l[m]) if m.any() else np.nan
    lo15, lo30 = low_by(15), low_by(30)
    for i in range(n):
        if mso[i] >= 15 and not pd.isna(lo15) and l[i] < lo15 and pd.isna(out["first_break_first15m_low_min"]):
            out["first_break_first15m_low_min"] = mso[i]
        if mso[i] >= 30 and not pd.isna(lo30) and l[i] < lo30 and pd.isna(out["first_break_first30m_low_min"]):
            out["first_break_first30m_low_min"] = mso[i]

    # swing highs/lows -> first *lower* swing high / *lower* swing low, and meaningful variants
    sh, sl = _swings(h, l, cfg.swing_lookback)
    meaningful = cfg.meaningful_move_atr_frac * (ref.get("atr") or np.nan)
    prev = None
    for idx in sh:
        if prev is not None and h[idx] < h[prev]:
            if pd.isna(out["first_lower_swing_high_min"]):
                out["first_lower_swing_high_min"] = mso[idx]
            if not pd.isna(meaningful) and (h[prev] - h[idx]) >= meaningful and pd.isna(out["first_meaningful_lower_high_min"]):
                out["first_meaningful_lower_high_min"] = mso[idx]
        prev = idx
    prev = None
    for idx in sl:
        if prev is not None and l[idx] < l[prev] and pd.isna(out["first_lower_swing_low_min"]):
            out["first_lower_swing_low_min"] = mso[idx]
        if prev is not None and l[idx] > l[prev] and not pd.isna(meaningful) and (l[idx] - l[prev]) >= meaningful and pd.isna(out["first_meaningful_higher_low_min"]):
            out["first_meaningful_higher_low_min"] = mso[idx]
        prev = idx

    # stall break: first close below the min-low of the preceding 3 bars (post first bar)
    for i in range(3, n):
        if c[i] < np.min(l[i - 3:i]):
            out["first_stall_break_min"] = mso[i]; break
    return out


# --------------------------------------------------------------------------- #
# per-session assembly
# --------------------------------------------------------------------------- #
def _ref_from_daily(row: pd.Series) -> dict:
    return {
        "cash_open": row["cash_open_adj"], "pdh": row["previous_day_high"],
        "pdl": row["previous_day_low"], "prev_close": row["previous_close"],
        "overnight_high": row["overnight_high_adj"], "overnight_low": row["overnight_low_adj"],
        "prior_ath": row["prior_ath"], "atr": row["atr"],
        "prev_day_range": row["previous_day_range"],
    }


def session_features(bars: pd.DataFrame, ref: dict, cfg: Config,
                     cutoff_min: int) -> dict:
    """All opening-pattern features for one session at a given cutoff."""
    b = bars[(bars["mso"] >= 0) & (bars["mso"] <= cutoff_min)].sort_values("mso")
    o = b["adj_open"].to_numpy(); h = b["adj_high"].to_numpy()
    l = b["adj_low"].to_numpy(); c = b["adj_close"].to_numpy()
    mso = b["mso"].to_numpy()
    feats = {"cutoff_min": cutoff_min, "n_window_bars": len(o)}
    if len(o) == 0:
        return feats
    feats.update(window_structure(o, h, l, c, mso, ref))
    feats.update(first_bar_features(o, h, l, c, ref))
    feats.update(post_spike_features(o, h, l, c, mso, ref, cfg))
    feats.update(weakness_features(o, h, l, c, mso, ref, cfg))
    # per fixed window: return / range / structure subset
    for w in cfg.opening_windows_min:
        bw = b[b["mso"] < w]
        if len(bw) == 0:
            continue
        ws = window_structure(bw["adj_open"].to_numpy(), bw["adj_high"].to_numpy(),
                              bw["adj_low"].to_numpy(), bw["adj_close"].to_numpy(),
                              bw["mso"].to_numpy(), ref)
        for k in ("net_return_pct", "high_low_range", "directional_efficiency",
                  "bar_overlap_ratio", "percentage_bull_bars",
                  "close_location_in_window_range"):
            feats[f"w{w}_{k}"] = ws.get(k)
    return feats


def build_opening_path_features(master: pd.DataFrame, daily: pd.DataFrame,
                                cfg: Config = DEFAULT_CONFIG,
                                cutoff_min: int | None = None,
                                sessions: list | None = None) -> pd.DataFrame:
    """Opening-pattern feature table (one row per session) at a cutoff."""
    if cutoff_min is None:
        cutoff_min = cfg.minutes_since_open(cfg.cutoff_time(cfg.default_cutoff))
    ms = sb.attach_sessions(master, cfg)
    maxw = max(cutoff_min, max(cfg.opening_windows_min))
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0) & (ms["mso"] <= maxw)]
    groups = dict(tuple(cash.groupby("session_date")))
    daily_idx = daily.set_index("session_date")
    rows = []
    dates = sessions if sessions is not None else list(groups.keys())
    for sd in dates:
        if sd not in groups or sd not in daily_idx.index:
            continue
        ref = _ref_from_daily(daily_idx.loc[sd])
        if pd.isna(ref["cash_open"]):
            continue
        f = session_features(groups[sd], ref, cfg, cutoff_min)
        f["session_date"] = sd
        rows.append(f)
    out = pd.DataFrame(rows)
    if "session_date" in out:
        cols = ["session_date"] + [c for c in out.columns if c != "session_date"]
        out = out[cols].sort_values("session_date").reset_index(drop=True)
    return out
