"""CLI for a price-action setup scan + conditional-probability report.

    python -m dax_analog_explorer.setup_report                      # the ATH-trap default
    python -m dax_analog_explorer.setup_report --gap-max 0.9 --close-beyond-prior-bar
    python -m dax_analog_explorer.setup_report --dates-only
    python -m dax_analog_explorer.setup_report --csv out.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import Config
from . import price_action as pa


def build_spec(a) -> pa.PriceActionSpec:
    return pa.PriceActionSpec(
        ath_tolerance_pct=a.ath_tol, require_new_ath_at_open=a.new_ath,
        require_open_above_pdh=not a.no_pdh, require_open_below_pdl=a.below_pdl,
        gap_min_pct=a.gap_min, gap_max_pct=a.gap_max,
        on_location_min=a.on_loc_min, on_location_max=a.on_loc_max,
        pressure_window_min=a.window, direction=a.direction,
        require_close_beyond_open=not a.no_close_beyond_open,
        close_location_max=a.close_loc,
        require_close_beyond_prior_bar=a.close_beyond_prior_bar,
        require_monotonic_highs=a.monotonic,
        min_directional_bars=a.min_bars,
        require_beyond_open_excursion=a.poke_reject,
        decision_bar=a.decision_bar, bar_minutes=a.bar_minutes,
        morning_end_min=a.morning_end)


def main():
    p = argparse.ArgumentParser(description="Price-action setup scan + conditional probabilities")
    p.add_argument("--ath-tol", type=float, default=0.50, dest="ath_tol")
    p.add_argument("--new-ath", action="store_true")
    p.add_argument("--no-pdh", action="store_true", help="drop the open>PDH condition")
    p.add_argument("--below-pdl", action="store_true")
    p.add_argument("--gap-min", type=float, default=0.30, dest="gap_min")
    p.add_argument("--gap-max", type=float, default=1.20, dest="gap_max")
    p.add_argument("--on-loc-min", type=float, default=0.70, dest="on_loc_min")
    p.add_argument("--on-loc-max", type=float, default=None, dest="on_loc_max")
    p.add_argument("--window", type=int, default=15, help="pressure window in minutes")
    p.add_argument("--direction", choices=["down", "up"], default="down")
    p.add_argument("--no-close-beyond-open", action="store_true")
    p.add_argument("--close-loc", type=float, default=0.33, dest="close_loc")
    p.add_argument("--close-beyond-prior-bar", action="store_true")
    p.add_argument("--monotonic", action="store_true")
    p.add_argument("--min-bars", type=int, default=None, dest="min_bars")
    p.add_argument("--poke-reject", action="store_true")
    p.add_argument("--decision-bar", type=int, default=6, dest="decision_bar")
    p.add_argument("--bar-minutes", type=int, default=5, dest="bar_minutes")
    p.add_argument("--morning-end", type=int, default=180, dest="morning_end")
    p.add_argument("--dates-only", action="store_true")
    p.add_argument("--rebuild", action="store_true",
                   help="recompute the cached session table")
    p.add_argument("--csv", default=None)
    p.add_argument("--json-spec", default=None, help="write the parameter set to a json file")
    a = p.parse_args()

    cfg = Config()
    spec = build_spec(a)
    daily = pd.read_parquet(cfg.paths.daily_features)
    master = pd.read_parquet(cfg.paths.master_5m)

    # the session table depends only on the TIMING parameters, so it can be
    # cached and reused across threshold tweaks.
    cache = cfg.paths.processed_dir / (
        f"session_table_w{spec.pressure_window_min}_b{spec.decision_bar}"
        f"_m{spec.morning_end_min}_{spec.direction}.parquet")
    if cache.exists() and not a.rebuild:
        table = pd.read_parquet(cache)
    else:
        table = pa.build_session_table(master, daily, spec, cfg)
        table.to_parquet(cache, index=False)
    matches, funnel = pa.scan(master, daily, spec, cfg, table=table)

    if a.dates_only:
        for d in matches["session_date"]:
            print(pd.Timestamp(d).date())
        return

    print("=" * 78)
    print("CONDITION CHAIN")
    print("=" * 78)
    for i, c in enumerate(spec.describe(), 1):
        print(f"  {i}. {c}")
    print(f"  → decide at bar {spec.decision_bar} "
          f"({spec.decision_min} min after the open)")

    print("\n" + "=" * 78)
    print("FUNNEL — sessions surviving each parameter")
    print("=" * 78)
    print(funnel.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"MATCHES — {len(matches)} sessions")
    print("=" * 78)
    if len(matches) == 0:
        print("  (none)")
        return
    print("  " + ", ".join(str(pd.Timestamp(d).date()) for d in matches["session_date"]))

    print("\n" + "=" * 78)
    print("CONDITIONAL PROBABILITY — P(level reached | untested at the decision bar)")
    print("=" * 78)
    print(pa.conditional_report(matches, spec).to_string(index=False))

    print("\n" + "=" * 78)
    print("PERFORMANCE FROM THE DECISION BAR")
    print("=" * 78)
    print(pa.performance_summary(matches).to_string(index=False))

    if len(matches) < 20:
        print(f"\n  ! small sample ({len(matches)}) — treat every rate as indicative.")

    if a.csv:
        matches.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")
    if a.json_spec:
        with open(a.json_spec, "w") as f:
            json.dump(spec.to_dict(), f, indent=2)
        print(f"wrote {a.json_spec}")


if __name__ == "__main__":
    main()
