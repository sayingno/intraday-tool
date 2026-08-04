"""Command-line preprocessing pipeline.

    python -m dax_analog_explorer.preprocess [--cutoff 10:30] [--no-audit]

Runs the whole chain once and writes the cleaned Parquet artifacts the Streamlit
app loads:

    cleaned_dax_5m.parquet         DAX 5m, Chicago->Berlin, deduped
    cleaned_fdax_1m.parquet        FDAX 1m, Berlin, + back-adjusted adj_* columns
    master_5m.parquet              continuous 5m (raw + adj), the merged series
    daily_features.parquet         one row per session
    opening_path_features.parquet  opening-pattern features at the default cutoff
    audit_report.json / .txt       data-quality report
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import data_loader as dl
from . import data_audit as da
from . import session_builder as sb
from . import feature_engineering as fe
from . import pattern_features as pf


def _log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def run(cfg: Config = DEFAULT_CONFIG, cutoff: str | None = None,
        do_audit: bool = True) -> dict:
    t0 = time.time()
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    _log("loading DAX 5m (Chicago->Berlin) ...", t0)
    dax = dl.load_dax_5m(cfg)
    _log(f"  DAX rows={len(dax.df)} dupes_dropped={dax.n_exact_duplicates}", t0)
    _log("loading FDAX 1m (Berlin) ...", t0)
    fdax = dl.load_fdax_1m(cfg)
    _log(f"  FDAX rows={len(fdax.df)} contracts={fdax.df['contract_month'].nunique()}", t0)

    if do_audit:
        _log("auditing ...", t0)
        report = da.run_full_audit(cfg, save=True)
        for w in report["warnings"]:
            print("   ! " + w)

    _log("building continuous master (roll back-adjustment: "
         f"{cfg.roll_adjust_method}) ...", t0)
    master, roll = sb.build_master_5m(dax.df, fdax.df, cfg)
    _log(f"  master rows={len(master)}  verification={roll.verification.get('note','')[:60]}", t0)

    # back-adjusted columns for the cleaned FDAX 1m (what the user asked for)
    fd = fdax.df.copy()
    is_dax = pd.Series(False, index=fd.index)
    for c in ("open", "high", "low", "close"):
        fd[f"adj_{c}"] = roll.adjust(fd[c], fd["con_id"], is_dax)

    _log("building daily features ...", t0)
    calendar = sb.build_calendar(sb.attach_sessions(master, cfg), cfg)
    daily = fe.build_daily_features(master, cfg, calendar)
    _log(f"  daily rows={len(daily)}", t0)

    cut_str = cutoff or cfg.default_cutoff
    cut_min = cfg.minutes_since_open(cfg.cutoff_time(cut_str))
    _log(f"building opening-path features at cutoff {cut_str} ({cut_min}m) ...", t0)
    opf = pf.build_opening_path_features(master, daily, cfg, cutoff_min=cut_min)
    _log(f"  opening rows={len(opf)}", t0)

    _log("writing parquet artifacts ...", t0)
    dax.df.to_parquet(cfg.paths.cleaned_dax_5m, index=False)
    fd.to_parquet(cfg.paths.cleaned_fdax_1m, index=False)
    master.to_parquet(cfg.paths.master_5m, index=False)
    daily.to_parquet(cfg.paths.daily_features, index=False)
    opf.to_parquet(cfg.paths.opening_path_features, index=False)
    _log("done.", t0)

    return {"dax": dax.df, "fdax": fd, "master": master, "daily": daily,
            "opening": opf, "roll": roll, "calendar": calendar}


def main():
    ap = argparse.ArgumentParser(description="DAX Analog Explorer preprocessing")
    ap.add_argument("--cutoff", default=None, help='observation cutoff e.g. "10:30"')
    ap.add_argument("--roll-method", default=None,
                    choices=["carry", "ratio", "difference", "none"])
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.roll_method:
        cfg.roll_adjust_method = args.roll_method
    run(cfg, cutoff=args.cutoff, do_audit=not args.no_audit)


if __name__ == "__main__":
    main()
