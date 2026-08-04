"""Human-facing reporting: interpreted query, distribution stats, dates-only.

The tool never claims analogs *predict* the current day -- reports always frame
outputs as a historical conditional distribution and always show sample size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .similarity_engine import FilterSpec, MatchTiers


def interpreted_query(spec: FilterSpec, cutoff: str, period_label: str,
                      n: int, mode: str) -> str:
    lines = [f"**Mode:** {mode}",
             f"**Historical period:** {period_label}",
             f"**Observation cutoff:** {cutoff} Berlin (no data after this time is used for matching)",
             f"**Results requested:** {n}",
             "**Conditions:**"]
    conds = spec.describe()
    if conds:
        lines += [f"- {c}" for c in conds]
    else:
        lines.append("- (none — pure similarity ranking)")
    return "\n".join(lines)


def relaxation_note(tiers: MatchTiers) -> str:
    c = tiers.counts
    out = [f"Exact matches: **{c.get('exact', 0)}**"]
    if tiers.relaxations_strong:
        out.append(f"Strong matches: **{c.get('strong', 0)}**")
        out += [f"  - {r}" for r in tiers.relaxations_strong]
    if tiers.relaxations_broad:
        out.append(f"Broad matches: **{c.get('broad', 0)}**")
        out += [f"  - {r}" for r in tiers.relaxations_broad]
    if c.get("exact", 0) < 5:
        out.append("\n> ⚠️ Very few exact cases — treat the distribution as indicative, not reliable.")
    return "\n".join(out)


OUTCOME_STAT_COLS = {
    "return_after_15m_pct": "Return +15m (%)",
    "return_after_30m_pct": "Return +30m (%)",
    "return_after_60m_pct": "Return +60m (%)",
    "return_at_1200_pct": "Return @12:00 (%)",
    "return_at_us_open_pct": "Return @US open (%)",
    "return_at_cash_close_pct": "Return @cash close (%)",
    "max_favorable_excursion_pct": "MFE (%)",
    "max_adverse_excursion_pct": "MAE (%)",
}


def distribution_stats(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Distributions, not just averages (spec §5)."""
    rows = []
    for col, label in OUTCOME_STAT_COLS.items():
        if col not in outcomes:
            continue
        v = outcomes[col].dropna().to_numpy()
        if len(v) == 0:
            continue
        rows.append({
            "metric": label, "n": len(v),
            "mean": np.mean(v), "median": np.median(v), "std": np.std(v),
            "positive_%": 100.0 * np.mean(v > 0),
            "p25": np.percentile(v, 25), "p75": np.percentile(v, 75),
            "best": np.max(v), "worst": np.min(v),
        })
    return pd.DataFrame(rows)


def outcome_headline(outcomes: pd.DataFrame) -> dict:
    def med(c):
        return float(outcomes[c].median()) if c in outcomes and outcomes[c].notna().any() else np.nan
    def rate(c):
        return float(100.0 * outcomes[c].mean()) if c in outcomes and outcomes[c].notna().any() else np.nan
    return {
        "sample_size": int(len(outcomes)),
        "median_return_at_close_pct": med("return_at_cash_close_pct"),
        "positive_close_%": rate("close_above_cash_open"),
        "median_MFE_pct": med("max_favorable_excursion_pct"),
        "median_MAE_pct": med("max_adverse_excursion_pct"),
        "full_gap_fill_%": rate("full_gap_fill"),
        "break_observed_high_%": rate("break_observed_high"),
        "break_observed_low_%": rate("break_observed_low"),
    }


def dates_only(dates) -> pd.DataFrame:
    d = [pd.Timestamp(x).date().isoformat() for x in dates]
    return pd.DataFrame({"date": d})


def similarity_explanation_text(explanation: dict) -> str:
    def fmt(xs):
        return ", ".join(xs) if xs else "—"
    return (f"**Highly similar:** {fmt(explanation.get('highly_similar'))}\n\n"
            f"**Moderately similar:** {fmt(explanation.get('moderately_similar'))}\n\n"
            f"**Differed:** {fmt(explanation.get('differed'))}")
