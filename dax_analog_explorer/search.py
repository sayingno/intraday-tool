"""Orchestration: turn a reference date + filters into ranked analogs + outcomes.

Keeps the leakage discipline end-to-end: matching features and paths are built
only from [open, cutoff]; outcomes are computed separately and never fed back
into ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import pattern_features as pf
from . import path_matching as pm
from . import outcome_engine as oe
from . import classifications as cl
from . import reports as rp
from .similarity_engine import (SimilarityEngine, FilterSpec, apply_filters,
                                progressive_search, MatchTiers)

MODE_EXACT = "Mode 1 — exact filter"
MODE_SIM = "Mode 2 — similarity"
MODE_COMBINED = "Mode 3 — combined (default)"


@dataclass
class AnalogResult:
    reference_date: pd.Timestamp
    reference_desc: str
    cutoff_min: int
    cutoff_str: str
    mode: str
    interpreted: str
    tiers: MatchTiers
    ranked_top: pd.DataFrame
    merged_top: pd.DataFrame
    explanations: dict
    grid: np.ndarray
    path_mat: np.ndarray
    path_kept: list
    path_unit: str
    stats: pd.DataFrame
    headline: dict
    n_pool: int
    period_label: str


def describe_reference(row: pd.Series, cfg: Config) -> str:
    def g(k, d=np.nan):
        return row[k] if k in row and not pd.isna(row[k]) else d
    bits = []
    dath = g("distance_open_to_ath_pct")
    if not pd.isna(dath):
        where = "above" if dath > 0 else "below"
        atv = "new ATH at open" if g("ath_at_open", False) else f"{abs(dath):.2f}% {where} prior ATH"
        bits.append(f"open {atv}")
    if g("ath_reached_overnight", False):
        bits.append("overnight reached the prior ATH")
    gp = g("gap_pct")
    if not pd.isna(gp):
        bits.append(f"gap {gp:+.2f}%")
    if g("open_above_pdh", False):
        bits.append("open above PDH")
    elif g("open_below_pdl", False):
        bits.append("open below PDL")
    fbr = g("first_bar_return"); fbn = g("first_bar_range_normalized")
    if not pd.isna(fbr):
        d = "bullish" if fbr > 0 else "bearish"
        bits.append(f"first 5m bar {d} ({fbr:+.2f}%, {fbn:.2f}×ATR)" if not pd.isna(fbn)
                    else f"first 5m bar {d} ({fbr:+.2f}%)")
    ext = g("extension_ratio"); ov = g("post_spike_overlap_ratio"); eff = g("post_spike_efficiency")
    if not pd.isna(ext):
        bits.append(f"follow-through: extension {ext:.2f}, overlap {ov:.2f}, efficiency {eff:.2f}")
    lh = g("first_meaningful_lower_high_min"); be = g("first_bearish_expansion_min")
    wk = min([v for v in [lh, be] if not pd.isna(v)], default=np.nan)
    if not pd.isna(wk):
        bits.append(f"first bearish weakness ~{wk:.0f}m after open")
    return "; ".join(bits)


def _period_bounds(daily: pd.DataFrame, period: str):
    end = daily["session_date"].max()
    table = {"Last year": 365, "Last 5 years": 5 * 365 + 1,
             "Last 10 years": 10 * 365 + 2, "Entire dataset": None}
    days = table.get(period, None)
    start = daily["session_date"].min() if days is None else end - pd.Timedelta(days=days)
    return start, end


def run_search(daily: pd.DataFrame, opening: pd.DataFrame | None, master: pd.DataFrame,
               *, reference_date, cutoff_min: int, cutoff_str: str, mode: str,
               spec: FilterSpec, weights: dict | None, period: str, n: int,
               path_method: str = "correlation", path_unit: str = "atr",
               cfg: Config = DEFAULT_CONFIG) -> AnalogResult:
    reference_date = pd.Timestamp(reference_date).normalize()
    start, end = _period_bounds(daily, period)

    # sessions in scope (cutoff-independent), always include the reference
    pdaily = daily[(daily["session_date"] >= start) & (daily["session_date"] <= end)]
    sess = sorted(set(pdaily["session_date"]).union({reference_date}))

    # opening-pattern features at the requested cutoff
    if opening is not None and int(opening["cutoff_min"].iloc[0]) == int(cutoff_min):
        op = opening[opening["session_date"].isin(sess)]
    else:
        op = pf.build_opening_path_features(master, daily, cfg, cutoff_min=cutoff_min,
                                            sessions=sess)

    m = daily[daily["session_date"].isin(sess)].merge(
        op.drop(columns=[c for c in ("weekday",) if c in op]),
        on="session_date", how="inner")
    m["day_net_points"] = m["cash_close_adj"] - m["cash_open_adj"]
    m["day_net_pct"] = 100.0 * m["day_net_points"] / m["cash_open_adj"]
    period_label = f"{period} ({start.date()} .. {end.date()}, {len(m)} sessions)"

    # ----- pool selection by mode -----
    if mode == MODE_EXACT:
        tiers = progressive_search(m, spec, cfg, min_sample=n)
        pool = list(m.loc[apply_filters(m, spec, cfg), "session_date"])
    elif mode == MODE_SIM:
        tiers = MatchTiers(exact=list(m["session_date"]),
                           counts={"exact": len(m)})
        pool = list(m["session_date"])
    else:  # combined (default): context pool + similarity ranking
        ctx = spec.context_only()
        tiers = progressive_search(m, ctx, cfg, min_sample=max(n, cfg.min_sample))
        pool = tiers.strong or tiers.broad or tiers.exact
    pool = [d for d in pool if d != reference_date]

    # ----- path matrix + distances to the reference -----
    refs = {r.session_date: {"cash_open": r.cash_open_adj,
                             "prev_day_range": r.previous_day_range, "atr": r.atr}
            for r in m.itertuples()}
    groups = pm.build_cash_groups(master, cfg, max_min=cutoff_min)
    grid, mat, kept = pm.build_path_matrix(groups, [reference_date] + pool,
                                           cutoff_min, path_unit, refs)
    pdist = {}
    if reference_date in kept:
        qi = kept.index(reference_date)
        for i, d in enumerate(kept):
            if d != reference_date:
                pdist[d] = pm.path_distance(mat[qi], mat[i], path_method)

    # ----- rank -----
    eng = SimilarityEngine(m, cfg, weights, cutoff_min)
    ranked = eng.rank(reference_date, pool, path_dist=pdist)
    top = ranked.head(n).copy()

    # ----- outcomes + classification + explanations -----
    outcomes = oe.build_outcomes(master, daily, cutoff_min, top["session_date"].tolist(), cfg)
    merged_top = top.merge(m, on="session_date", how="left").merge(
        outcomes, on="session_date", how="left", suffixes=("", "_out"))
    merged_top = cl.classify_frame(merged_top, cfg)
    explanations = {d: eng.explain(reference_date, d)
                    for d in top["session_date"] if d in eng.S.index}

    stats = rp.distribution_stats(merged_top)
    headline = rp.outcome_headline(merged_top)

    ref_row = m[m["session_date"] == reference_date]
    ref_desc = describe_reference(ref_row.iloc[0], cfg) if len(ref_row) else "(reference not in feature set)"
    interpreted = rp.interpreted_query(spec, cutoff_str, period_label, n, mode)

    return AnalogResult(
        reference_date=reference_date, reference_desc=ref_desc,
        cutoff_min=cutoff_min, cutoff_str=cutoff_str, mode=mode,
        interpreted=interpreted, tiers=tiers, ranked_top=top, merged_top=merged_top,
        explanations=explanations, grid=grid, path_mat=mat, path_kept=kept,
        path_unit=path_unit, stats=stats, headline=headline,
        n_pool=len(pool), period_label=period_label)


# --------------------------------------------------------------------------- #
# helpers for the UI / charts
# --------------------------------------------------------------------------- #
def day_bars(master: pd.DataFrame, date, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Full cash-session bars for one date, with mso, for charting."""
    from . import session_builder as sb
    ms = sb.attach_sessions(master, cfg)
    date = pd.Timestamp(date).normalize()
    return ms[(ms["session_date"] == date) & ms["is_cash"] & (ms["mso"] >= 0)].sort_values("mso")


def flagship_filter(cfg: Config = DEFAULT_CONFIG) -> FilterSpec:
    """The worked example's filter (ATH gap-up -> weak follow-through -> weakness)."""
    return FilterSpec(
        ath_within_pct=cfg.default_ath_tolerance_pct, gap_sign="positive",
        open_above_pdh=True, first_bar_bull=True,
        first_bar_range_pctl_min=cfg.first_bar_range_pctl,
        first_bar_close_top_frac=cfg.first_bar_close_top_frac,
        extension_ratio_max=cfg.weak_extension_ratio_max,
        overlap_ratio_min=cfg.weak_overlap_ratio_min,
        weakness_before_min=cfg.minutes_since_open(cfg.cutoff_time(cfg.weakness_deadline)))
