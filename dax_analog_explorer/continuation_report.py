"""CLI: the opening trend-continuation study, backtest and chart PDF.

    python -m dax_analog_explorer.continuation_report                 # full study
    python -m dax_analog_explorer.continuation_report --pdf out.pdf   # + chart pack
    python -m dax_analog_explorer.continuation_report --target 3 --trail 6
    python -m dax_analog_explorer.continuation_report --location BREAKOUT_UP
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import Config
from . import context as ctx
from . import continuation as co
from . import chart_pdf as cp


def main():
    ap = argparse.ArgumentParser(description="Opening trend-continuation study")
    ap.add_argument("--decision-min", type=int, default=30, dest="decision_min")
    ap.add_argument("--stop", default="opening_range",
                    choices=["opening_range", "half_range", "last_bar"])
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--no-target", action="store_true")
    ap.add_argument("--breakeven", type=float, default=None)
    ap.add_argument("--trail", type=int, default=None)
    ap.add_argument("--time-exit", type=int, default=None, dest="time_exit")
    ap.add_argument("--cost", type=float, default=2.0)
    ap.add_argument("--side", choices=["both", "up", "down"], default="both")
    ap.add_argument("--split", default="2017-01-01")
    # optional context narrowing
    ap.add_argument("--location", nargs="*", default=None)
    ap.add_argument("--swing", nargs="*", default=None)
    ap.add_argument("--gap", nargs="*", default=None)
    ap.add_argument("--prior-day", nargs="*", default=None, dest="prior_day")
    ap.add_argument("--pdf", default=None, help="write a chart pack to this path")
    ap.add_argument("--pdf-max", type=int, default=40, dest="pdf_max")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    cfg = Config()
    daily = pd.read_parquet(cfg.paths.daily_features)
    master = pd.read_parquet(cfg.paths.master_5m)
    tpath = cfg.paths.processed_dir / "context_table.parquet"
    if tpath.exists():
        T = pd.read_parquet(tpath)
    else:
        T = ctx.build_context_table(master, daily, cfg, decision_min=a.decision_min)
        T.to_parquet(tpath, index=False)

    # narrow by context if asked
    spec = ctx.ContextSpec(
        location=tuple(a.location) if a.location else None,
        swing=tuple(a.swing) if a.swing else None,
        gap_bucket=tuple(a.gap) if a.gap else None,
        prior_day_type=tuple(a.prior_day) if a.prior_day else None)
    Tn = T[ctx.apply_context(T, spec)] if any(
        getattr(spec, f) for f in ctx.ContextSpec.FIELDS) else T

    labels = {"both": ("DRIVE_UP", "DRIVE_DOWN"),
              "up": ("DRIVE_UP",), "down": ("DRIVE_DOWN",)}[a.side]
    rules = co.TradeRules(decision_min=a.decision_min, stop_mode=a.stop,
                          target_R=(None if a.no_target else a.target),
                          breakeven_at_R=a.breakeven, trail_bars=a.trail,
                          time_exit_min=a.time_exit, cost_points=a.cost)

    print("=" * 82)
    print("STRATEGY")
    print("=" * 82)
    for i, s in enumerate(rules.describe(), 1):
        print(f"  {i}. {s}")
    if any(getattr(spec, f) for f in ctx.ContextSpec.FIELDS):
        print("  context:")
        for s in spec.describe():
            print(f"     · {s}")
    print(f"  side: {a.side}")

    trades = co.build_trades(master, Tn, rules, cfg, labels=labels)
    if len(trades) == 0:
        print("\nno trades matched")
        return

    print("\n" + "=" * 82)
    print(f"RESULTS — {len(trades)} trades")
    print("=" * 82)
    print(co.split_performance(trades, a.split).to_string(index=False))

    print("\nexit reasons:")
    for k, v in trades["reason"].value_counts().items():
        sub = trades[trades["reason"] == k]["R_multiple"]
        print(f"   {k:<10}{v:>5}  ({100*v/len(trades):>4.1f}%)   mean {sub.mean():+.2f}R")

    r = trades["R_multiple"]
    print(f"\nR distribution: p10 {r.quantile(.10):+.2f}  p25 {r.quantile(.25):+.2f}  "
          f"median {r.median():+.2f}  p75 {r.quantile(.75):+.2f}  p90 {r.quantile(.90):+.2f}")
    print(f"best {r.max():+.2f}R   worst {r.min():+.2f}R")

    perf = co.performance(trades)
    if abs(perf["t_stat"]) < 2:
        print(f"\n  ! expectancy {perf['expectancy_R']:+.3f}R ± {perf['se_R']:.3f} "
              f"(t={perf['t_stat']}) is NOT distinguishable from zero.")

    if a.csv:
        trades.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")
    if a.pdf:
        dates = trades["session_date"].tolist()[-a.pdf_max:]
        cp.sessions_to_pdf(a.pdf, dates, master, daily, decision_min=a.decision_min,
                           trades=trades, context=T, cfg=cfg,
                           title=f"Opening trend continuation — {len(dates)} sessions")
        print(f"wrote {a.pdf}  ({len(dates)} sessions)")


if __name__ == "__main__":
    main()
