"""Similarity + search engine.

Three modes (spec):
  Mode 1  exact condition filter               -> every date passing the filter
  Mode 2  similarity ranking                   -> all eligible ranked by closeness
  Mode 3  combined (default)                    -> broad filter, then rank

Scaling is robust (median / IQR) so fat tails and outliers do not dominate.
Distance is a weighted blend of four feature blocks -- market context (30%),
first-bar structure (20%), post-spike path (30%, into which normalized path
correlation is folded), weakness structure (20%) -- every weight configurable.

Progressive matching keeps sample size honest: it widens tolerances along a
recorded ladder until enough analogs exist, and every relaxation is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG


FEATURE_LABELS = {
    "distance_open_to_ath_pct": "distance of open to prior ATH",
    "overnight_return_pct": "overnight return", "overnight_range_normalized": "overnight range (ATR)",
    "gap_pct": "opening gap", "open_vs_pdh_pct": "open vs previous-day high",
    "first_bar_return": "first-bar return", "first_bar_range_normalized": "first-bar size (ATR)",
    "first_bar_body_to_range": "first-bar body/range", "first_bar_close_location": "first-bar close location",
    "first_bar_breaks_overnight_high": "first bar breaks overnight high",
    "extension_ratio": "post-spike extension", "post_spike_pullback_ratio": "post-spike pullback",
    "post_spike_overlap_ratio": "post-spike overlap", "post_spike_efficiency": "post-spike efficiency",
    "number_of_new_highs_after_first_bar": "new highs after first bar",
    "first_meaningful_lower_high_min": "time of first lower high",
    "first_bearish_expansion_min": "time of first bearish expansion",
    "first_break_first15m_low_min": "time breaking first-15m low",
    "time_below_open_frac": "time spent below the open",
}


# --------------------------------------------------------------------------- #
# exact filters (Mode 1)
# --------------------------------------------------------------------------- #
@dataclass
class FilterSpec:
    ath_within_pct: float | None = None        # |open-ATH|/ATH <= x
    require_ath_at_open: bool = False
    require_ath_overnight: bool = False
    gap_sign: str | None = None                # "positive" / "negative"
    gap_min_pct: float | None = None
    open_above_pdh: bool = False
    open_below_pdl: bool = False
    vol_regimes: tuple[str, ...] | None = None
    first_bar_bull: bool = False
    first_bar_range_pctl_min: float | None = None   # >= this historical percentile
    first_bar_close_top_frac: float | None = None   # close location >= x
    extension_ratio_max: float | None = None
    overlap_ratio_min: float | None = None
    efficiency_max: float | None = None
    pullback_ratio_min: float | None = None
    weakness_before_min: float | None = None        # any weakness event by this minute

    def context_only(self) -> "FilterSpec":
        """Copy keeping only market-context gates (Mode 3 pool); pattern-structure
        conditions become ranking preferences, not hard filters."""
        return FilterSpec(
            ath_within_pct=self.ath_within_pct,
            require_ath_at_open=self.require_ath_at_open,
            require_ath_overnight=self.require_ath_overnight,
            gap_sign=self.gap_sign, gap_min_pct=self.gap_min_pct,
            open_above_pdh=self.open_above_pdh, open_below_pdl=self.open_below_pdl,
            vol_regimes=self.vol_regimes)

    def describe(self) -> list[str]:
        d = []
        if self.ath_within_pct is not None: d.append(f"within {self.ath_within_pct:g}% of prior ATH")
        if self.require_ath_at_open: d.append("open above prior ATH")
        if self.require_ath_overnight: d.append("overnight reached prior ATH")
        if self.gap_sign: d.append(f"{self.gap_sign} gap")
        if self.gap_min_pct is not None: d.append(f"|gap| >= {self.gap_min_pct:g}%")
        if self.open_above_pdh: d.append("open above PDH")
        if self.open_below_pdl: d.append("open below PDL")
        if self.vol_regimes: d.append(f"volatility regime in {list(self.vol_regimes)}")
        if self.first_bar_bull: d.append("bullish first bar")
        if self.first_bar_range_pctl_min is not None: d.append(f"first-bar range >= {self.first_bar_range_pctl_min:g}th pctl")
        if self.first_bar_close_top_frac is not None: d.append(f"first-bar close in top {100*(1-self.first_bar_close_top_frac):g}%")
        if self.extension_ratio_max is not None: d.append(f"extension ratio < {self.extension_ratio_max:g}")
        if self.overlap_ratio_min is not None: d.append(f"overlap ratio > {self.overlap_ratio_min:g}")
        if self.efficiency_max is not None: d.append(f"post-spike efficiency < {self.efficiency_max:g}")
        if self.pullback_ratio_min is not None: d.append(f"post-spike pullback ratio > {self.pullback_ratio_min:g}")
        if self.weakness_before_min is not None: d.append(f"bearish weakness before {self.weakness_before_min:g}m after open")
        return d


WEAKNESS_TIME_COLS = ["first_meaningful_lower_high_min", "first_bearish_expansion_min",
                      "first_stall_break_min", "first_break_first15m_low_min",
                      "first_lower_swing_high_min", "first_close_below_prev2_low_min"]


def apply_filters(m: pd.DataFrame, spec: FilterSpec, cfg: Config = DEFAULT_CONFIG) -> pd.Series:
    """Boolean mask over merged daily+pattern rows for the exact filter."""
    mask = pd.Series(True, index=m.index)
    if spec.ath_within_pct is not None:
        mask &= m["distance_open_to_ath_pct"].abs() <= spec.ath_within_pct
    if spec.require_ath_at_open:
        mask &= m["ath_at_open"].fillna(False)
    if spec.require_ath_overnight:
        mask &= m["ath_reached_overnight"].fillna(False)
    if spec.gap_sign == "positive":
        mask &= m["gap_pct"] > 0
    elif spec.gap_sign == "negative":
        mask &= m["gap_pct"] < 0
    if spec.gap_min_pct is not None:
        mask &= m["gap_pct"].abs() >= spec.gap_min_pct
    if spec.open_above_pdh:
        mask &= m["open_above_pdh"].fillna(False)
    if spec.open_below_pdl:
        mask &= m["open_below_pdl"].fillna(False)
    if spec.vol_regimes:
        mask &= m["daily_volatility_regime"].isin(spec.vol_regimes)
    if spec.first_bar_bull:
        mask &= m["first_bar_is_bull"].fillna(False)
    if spec.first_bar_range_pctl_min is not None and "first_bar_range_normalized" in m:
        thr = np.nanpercentile(m["first_bar_range_normalized"].dropna(), spec.first_bar_range_pctl_min)
        mask &= m["first_bar_range_normalized"] >= thr
    if spec.first_bar_close_top_frac is not None:
        mask &= m["first_bar_close_location"] >= spec.first_bar_close_top_frac
    if spec.extension_ratio_max is not None:
        mask &= m["extension_ratio"] < spec.extension_ratio_max
    if spec.overlap_ratio_min is not None:
        mask &= m["post_spike_overlap_ratio"] > spec.overlap_ratio_min
    if spec.efficiency_max is not None:
        mask &= m["post_spike_efficiency"] < spec.efficiency_max
    if spec.pullback_ratio_min is not None:
        mask &= m["post_spike_pullback_ratio"] > spec.pullback_ratio_min
    if spec.weakness_before_min is not None:
        cols = [c for c in WEAKNESS_TIME_COLS if c in m]
        early = pd.Series(False, index=m.index)
        for c in cols:
            early |= m[c] <= spec.weakness_before_min
        mask &= early
    return mask.fillna(False)


# --------------------------------------------------------------------------- #
# similarity engine (Mode 2 / 3)
# --------------------------------------------------------------------------- #
class SimilarityEngine:
    def __init__(self, feats: pd.DataFrame, cfg: Config = DEFAULT_CONFIG,
                 weights: dict | None = None, cutoff_min: int | None = None):
        self.cfg = cfg
        self.cutoff_min = cutoff_min or cfg.minutes_since_open(cfg.cutoff_time())
        self.feats = feats.drop_duplicates("session_date").set_index("session_date").sort_index()
        self.blocks = {
            "context": list(cfg.context_features),
            "first_bar": list(cfg.first_bar_features),
            "post_spike": list(cfg.post_spike_features),
            "weakness": list(cfg.weakness_features),
        }
        w = weights or {"context": cfg.weight_context, "first_bar": cfg.weight_first_bar,
                        "post_spike": cfg.weight_post_spike, "weakness": cfg.weight_weakness}
        s = sum(w.values()) or 1.0
        self.block_weights = {k: v / s for k, v in w.items()}
        self.all_feats = [f for b in self.blocks.values() for f in b if f in self.feats.columns]
        self._fit()

    def _impute(self, col: str, s: pd.Series) -> pd.Series:
        s = s.astype("float64")
        if col.endswith("_min"):
            fill = self.cutoff_min + 30          # "did not happen" -> late/never
        else:
            fill = s.median()
        return s.fillna(fill)

    def _fit(self):
        med, iqr, scaled = {}, {}, {}
        for c in self.all_feats:
            col = self.feats[c]
            if col.dtype == bool:
                col = col.astype("float64")
            col = self._impute(c, col)
            m = col.median()
            q1, q3 = col.quantile(0.25), col.quantile(0.75)
            spread = (q3 - q1) or col.std() or 1.0
            med[c] = m; iqr[c] = spread
            scaled[c] = (col - m) / spread
        self.median, self.iqr = med, iqr
        self.S = pd.DataFrame(scaled, index=self.feats.index)

    def _block_distance(self, block: str, qvec: pd.Series, cand: pd.Index) -> np.ndarray:
        cols = [c for c in self.blocks[block] if c in self.S.columns]
        if not cols:
            return np.zeros(len(cand))
        A = self.S.loc[cand, cols].to_numpy()
        q = qvec[cols].to_numpy().reshape(1, -1)
        return np.sqrt(np.nanmean((A - q) ** 2, axis=1))

    def rank(self, query_date, candidate_dates, path_dist: dict | None = None) -> pd.DataFrame:
        cand = pd.Index([d for d in candidate_dates
                         if d in self.S.index and d != query_date])
        if query_date not in self.S.index or len(cand) == 0:
            return pd.DataFrame(columns=["session_date", "similarity_score"])
        qvec = self.S.loc[query_date]
        block_d = {b: self._block_distance(b, qvec, cand) for b in self.blocks}

        # fold normalized path distance into the post-spike block
        if path_dist is not None:
            pv = np.array([path_dist.get(d, np.nan) for d in cand], dtype="float64")
            if np.isfinite(pv).any():
                med = np.nanmedian(pv); pv = np.where(np.isnan(pv), med, pv)
                pw = self.cfg.path_corr_weight_in_post
                block_d["post_spike"] = (1 - pw) * block_d["post_spike"] + pw * (pv / (np.nanmedian(pv) or 1.0))

        total = np.zeros(len(cand))
        for b, dist in block_d.items():
            total += self.block_weights[b] * dist
        score = 100.0 / (1.0 + total)
        out = pd.DataFrame({"session_date": cand, "similarity_score": score,
                            "distance": total})
        for b in self.blocks:
            out[f"dist_{b}"] = block_d[b]
        return out.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    # -- per-result explanation -------------------------------------------
    def explain(self, query_date, cand_date) -> dict:
        q = self.S.loc[query_date]; c = self.S.loc[cand_date]
        highly, moderately, differed = [], [], []
        for f in self.all_feats:
            diff = abs(q[f] - c[f])
            label = FEATURE_LABELS.get(f, f)
            if diff < 0.5:
                highly.append((label, diff))
            elif diff < 1.5:
                moderately.append((label, diff))
            else:
                differed.append((label, diff))
        highly.sort(key=lambda x: x[1]); moderately.sort(key=lambda x: x[1])
        differed.sort(key=lambda x: -x[1])
        return {
            "highly_similar": [l for l, _ in highly[:6]],
            "moderately_similar": [l for l, _ in moderately[:6]],
            "differed": [l for l, _ in differed[:6]],
        }


# --------------------------------------------------------------------------- #
# progressive matching
# --------------------------------------------------------------------------- #
@dataclass
class MatchTiers:
    exact: list = field(default_factory=list)
    strong: list = field(default_factory=list)
    broad: list = field(default_factory=list)
    relaxations_strong: list = field(default_factory=list)
    relaxations_broad: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def progressive_search(m: pd.DataFrame, base: FilterSpec, cfg: Config = DEFAULT_CONFIG,
                       min_sample: int | None = None) -> MatchTiers:
    """Widen the filter along the configured ladder until min_sample analogs."""
    min_sample = min_sample or cfg.min_sample
    exact_dates = m.loc[apply_filters(m, base, cfg), "session_date"].tolist()
    tiers = MatchTiers(exact=exact_dates)
    tiers.counts["exact"] = len(exact_dates)

    spec = FilterSpec(**{**base.__dict__})
    applied = []
    strong_done = False
    for step in cfg.relax_ladder:
        name = step["name"]
        if name == "ath_tolerance" and spec.ath_within_pct is not None:
            spec.ath_within_pct = step["to"]
        elif name == "weakness_deadline" and spec.weakness_before_min is not None:
            spec.weakness_before_min = _hhmm_to_min(step["to"], cfg)
        elif name == "drop_gap":
            spec.gap_sign = None; spec.gap_min_pct = None
        elif name == "drop_regime":
            spec.vol_regimes = None
        else:
            continue
        applied.append(step["desc"])
        dates = m.loc[apply_filters(m, spec, cfg), "session_date"].tolist()
        if not strong_done and len(dates) >= min_sample:
            tiers.strong = dates
            tiers.relaxations_strong = list(applied)
            tiers.counts["strong"] = len(dates)
            strong_done = True
        tiers.broad = dates
        tiers.relaxations_broad = list(applied)
        tiers.counts["broad"] = len(dates)
        if strong_done and len(dates) >= min_sample * 2:
            break
    if not tiers.strong:
        tiers.strong = tiers.broad or exact_dates
        tiers.counts.setdefault("strong", len(tiers.strong))
    return tiers


def _hhmm_to_min(s, cfg: Config) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    return float(cfg.minutes_since_open(cfg.cutoff_time(s)))
