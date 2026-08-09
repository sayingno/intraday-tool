"""CLI: category-based context scan + conditional probabilities.

    python -m dax_analog_explorer.context_report                     # the ATH bull trap
    python -m dax_analog_explorer.context_report --list              # show every category
    python -m dax_analog_explorer.context_report --location AT_ATH NEAR_ATH --opening REJECTION_UP
    python -m dax_analog_explorer.context_report --prior-day BIG_RANGE --gap SMALL_UP
    python -m dax_analog_explorer.context_report --profile           # label mix of the dataset
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import Config
from . import context as ctx
from . import price_action as pa

CATEGORIES = {
    "location": ctx.LOCATION, "swing": ctx.SWING, "ma_state": ctx.MA_STATE,
    "prior_day_type": ctx.PRIOR_DAY, "gap_bucket": ctx.GAP,
    "open_loc": ctx.OPEN_LOC, "overnight_type": ctx.OVERNIGHT, "opening": ctx.OPENING,
}


def _cache(cfg: Config, decision_min: int) -> pd.DataFrame:
    path = cfg.paths.processed_dir / f"context_table_b{decision_min}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    daily = pd.read_parquet(cfg.paths.daily_features)
    master = pd.read_parquet(cfg.paths.master_5m)
    t = ctx.build_context_table(master, daily, cfg, decision_min=decision_min)
    t.to_parquet(path, index=False)
    return t


def main():
    ap = argparse.ArgumentParser(description="Category-based context scan")
    for field in CATEGORIES:
        ap.add_argument("--" + field.replace("_", "-").replace("-type", ""),
                        nargs="*", dest=field, default=None)
    ap.add_argument("--decision-bar", type=int, default=6, dest="decision_bar")
    ap.add_argument("--bar-minutes", type=int, default=5, dest="bar_minutes")
    ap.add_argument("--morning-end", type=int, default=180, dest="morning_end")
    ap.add_argument("--list", action="store_true", help="print every category and exit")
    ap.add_argument("--profile", action="store_true", help="label mix of the dataset")
    ap.add_argument("--dates-only", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if a.list:
        print("Available categories — pass any subset:\n")
        for field, vals in CATEGORIES.items():
            flag = "--" + field.replace("_", "-").replace("-type", "")
            print(f"  {flag:<16} {' · '.join(vals)}")
        return

    cfg = Config()
    dm = a.decision_bar * a.bar_minutes
    table = _cache(cfg, dm)

    if a.profile:
        print(f"Label mix across {len(table)} sessions\n")
        for field in CATEGORIES:
            vc = table[field].value_counts()
            print(f"{field}:")
            for k, v in vc.items():
                print(f"   {k:<16}{v:>6}  ({100*v/len(table):>4.1f}%)")
            print()
        return

    spec = ctx.ContextSpec(**{f: (tuple(getattr(a, f)) if getattr(a, f) else None)
                              for f in CATEGORIES})
    if not any(getattr(spec, f) for f in ctx.ContextSpec.FIELDS):
        spec = ctx.ATH_TRAP_CONTEXT
        print("(no categories given — using the built-in ATH bull-trap spec)\n")

    print("=" * 78)
    print("CONTEXT")
    print("=" * 78)
    for i, c in enumerate(spec.describe(), 1):
        print(f"  {i}. {c}")
    print(f"  → decide at bar {a.decision_bar} ({dm} min after the open)")

    print("\n" + "=" * 78)
    print("FUNNEL")
    print("=" * 78)
    print(ctx.context_funnel(table, spec).to_string(index=False))

    matches = table[ctx.apply_context(table, spec)]
    if a.dates_only:
        for d in matches["session_date"]:
            print(pd.Timestamp(d).date())
        return

    print("\n" + "=" * 78)
    print(f"MATCHES — {len(matches)} sessions")
    print("=" * 78)
    if len(matches) == 0:
        print("  none — drop a category or widen one")
        return
    print("  " + ", ".join(str(pd.Timestamp(d).date()) for d in matches["session_date"]))

    # outcomes + conditional probabilities, reusing the price-action machinery
    pspec = pa.PriceActionSpec(
        ath_tolerance_pct=None, require_open_above_pdh=False, gap_min_pct=None,
        gap_max_pct=None, on_location_min=None, require_close_beyond_open=False,
        close_location_max=None, decision_bar=a.decision_bar,
        bar_minutes=a.bar_minutes, morning_end_min=a.morning_end)
    sess = pa.build_session_table(
        pd.read_parquet(cfg.paths.master_5m), pd.read_parquet(cfg.paths.daily_features),
        pspec, cfg)
    sel = sess[sess["session_date"].isin(matches["session_date"])]

    print("\n" + "=" * 78)
    print("CONDITIONAL PROBABILITY — P(level reached | untested at the decision bar)")
    print("=" * 78)
    print(pa.conditional_report(sel, pspec).to_string(index=False))

    print("\n" + "=" * 78)
    print("PERFORMANCE FROM THE DECISION BAR")
    print("=" * 78)
    print(pa.performance_summary(sel).to_string(index=False))
    if len(sel) < 20:
        print(f"\n  ! small sample ({len(sel)}) — treat every rate as indicative.")

    if a.csv:
        matches.merge(sel, on="session_date", how="left").to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
