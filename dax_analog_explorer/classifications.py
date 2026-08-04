"""Explicit, editable rules that label a session's overall shape.

Rules are intentionally transparent (priority-ordered if/elif) and every
threshold is overridable via ``ClassRules`` so the UI can expose them.  The
labels are the eight day-types from the spec.  Classification is descriptive
(post-hoc), never a signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG


LABELS = [
    "ATH gap-and-go continuation",
    "opening spike then bull flag",
    "opening spike failure",
    "partial gap fade",
    "full gap fill",
    "complete reversal",
    "high-level range day",
    "two-sided volatile day",
]


@dataclass
class ClassRules:
    big_first_bar_atr: float = 0.5      # first-bar range >= this * ATR  => "spike"
    weak_extension_ratio: float = 0.5   # extension < this               => weak follow-through
    high_overlap: float = 0.5           # overlap  > this                => congestion
    trend_net_atr: float = 0.75         # |day net| >= this * ATR        => directional
    range_net_atr: float = 0.4          # |day net| <  this * ATR        => rangey
    big_excursion_atr: float = 0.8      # excursion >= this * ATR        => "large"

    @classmethod
    def from_config(cls, cfg: Config) -> "ClassRules":
        return cls(weak_extension_ratio=cfg.weak_extension_ratio_max,
                   high_overlap=cfg.weak_overlap_ratio_min)


def _atrfrac(x, atr):
    if pd.isna(x) or pd.isna(atr) or atr == 0:
        return np.nan
    return x / atr


def classify_day(row: pd.Series, cfg: Config = DEFAULT_CONFIG,
                 rules: ClassRules | None = None) -> tuple[str, str]:
    """Return (label, human-readable reason)."""
    r = rules or ClassRules.from_config(cfg)
    atr = row.get("atr", np.nan)
    day_net = row.get("day_net_pct", np.nan)                # % close vs open
    day_net_pts = row.get("day_net_points", np.nan)
    gap = row.get("gap_pct", np.nan)
    mfe = row.get("max_favorable_excursion_pct", np.nan)
    mae = row.get("max_adverse_excursion_pct", np.nan)
    full_fill = bool(row.get("full_gap_fill", False))
    half_fill = bool(row.get("half_gap_fill", False))
    big_spike = _atrfrac(row.get("first_bar_range", np.nan), atr)
    big_spike = (big_spike is not np.nan) and (big_spike >= r.big_first_bar_atr)
    bull_first = bool(row.get("first_bar_is_bull", False))
    ext = row.get("extension_ratio", np.nan)
    overlap = row.get("post_spike_overlap_ratio", np.nan)
    weak_ft = (not pd.isna(ext) and ext < r.weak_extension_ratio) and \
              (not pd.isna(overlap) and overlap > r.high_overlap)
    weakness_min = min([row.get("first_meaningful_lower_high_min", np.nan),
                        row.get("first_bearish_expansion_min", np.nan),
                        row.get("first_stall_break_min", np.nan)],
                       key=lambda v: (np.inf if pd.isna(v) else v))
    dl = cfg.minutes_since_open(cfg.cutoff_time(cfg.weakness_deadline))
    early_weakness = (not pd.isna(weakness_min)) and (weakness_min <= dl)
    close_above_open = bool(row.get("close_above_cash_open", day_net_pts > 0))
    ath_ctx = bool(row.get("ath_at_open", False)) or bool(row.get("ath_reached_overnight", False)) \
        or bool(row.get("near_ath_0.5pct", False))
    net_atr = abs(_atrfrac(day_net_pts, atr)) if not pd.isna(day_net_pts) else np.nan
    two_sided = (not pd.isna(mfe) and not pd.isna(mae) and
                 mfe >= r.big_excursion_atr * (100 * atr / row.get("cash_open_adj", np.nan) if atr else np.nan) and
                 -mae >= r.big_excursion_atr * (100 * atr / row.get("cash_open_adj", np.nan) if atr else np.nan))

    # ---- priority-ordered decision ----
    if full_fill and not close_above_open and gap > 0:
        return LABELS[4], f"gap +{gap:.2f}% fully filled back to previous close; closed below open."
    if gap > 0 and not pd.isna(net_atr) and close_above_open is False and (mae is not np.nan and -mae > r.big_excursion_atr):
        return LABELS[5], f"opened with +{gap:.2f}% gap but reversed hard (MAE {mae:.2f}%) and closed weak."
    if ath_ctx and big_spike and bull_first and close_above_open and not weak_ft and \
            (not pd.isna(net_atr) and net_atr >= r.trend_net_atr):
        return LABELS[0], "ATH context, large bullish first bar, strong follow-through into the close."
    if big_spike and bull_first and early_weakness and (not close_above_open or (mae is not np.nan and -mae > r.big_excursion_atr)):
        return LABELS[2], f"large bullish spike then early weakness (~{weakness_min:.0f}m) that did not hold."
    if big_spike and bull_first and weak_ft and close_above_open:
        return LABELS[1], "bullish spike, weak overlapping follow-through, but held and drifted up (bull flag)."
    if gap > 0 and half_fill and not full_fill:
        return LABELS[3], f"gap +{gap:.2f}% partially faded (half-fill) without fully closing."
    if two_sided:
        return LABELS[7], "large excursions both ways -- whippy two-sided session."
    if not pd.isna(net_atr) and net_atr < r.range_net_atr:
        return LABELS[6], "small net change with price holding a high-level range."
    # fallback
    direction = "up" if (not pd.isna(day_net) and day_net > 0) else "down"
    return LABELS[6], f"contained {direction} session (default classification)."


def classify_frame(merged: pd.DataFrame, cfg: Config = DEFAULT_CONFIG,
                   rules: ClassRules | None = None) -> pd.DataFrame:
    labels, reasons = [], []
    for _, row in merged.iterrows():
        lab, why = classify_day(row, cfg, rules)
        labels.append(lab); reasons.append(why)
    out = merged.copy()
    out["day_classification"] = labels
    out["day_classification_reason"] = reasons
    return out
