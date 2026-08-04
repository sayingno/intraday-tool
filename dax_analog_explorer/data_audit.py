"""Data audit: report (never silently fix) the state of the raw inputs.

Produces a structured dict (also saved to JSON + a human-readable txt) covering:
detected timezone, coverage, session hours, missing/duplicate/gap counts,
contract identifiers, roll seams + the back-adjustment verification, and the
known cross-dataset gap.  Nothing here assumes Chicago or Berlin -- the timezone
is *detected* from the intraday volume profile and compared to the declared one.
"""
from __future__ import annotations

import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .data_loader import load_dax_5m, load_fdax_1m, LoadResult
from . import session_builder as sb


def _hhmm(m: int) -> str:
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def detect_timezone(df: pd.DataFrame, cfg: Config) -> dict:
    """Infer the session timezone from the volume profile and compare with the
    declared Europe/Berlin.  The open (09:00) and close (17:30) auctions are the
    fingerprints; if the data were mislabeled the peaks would be hours off.
    """
    b = df["dt"].dt.tz_convert(ZoneInfo(cfg.tz))
    tod = b.dt.hour * 60 + b.dt.minute
    prof = df.assign(tod=tod).groupby("tod")["volume"].sum()
    open_min = cfg.cash_open_minutes()
    close_min = cfg.cash_close.hour * 60 + cfg.cash_close.minute
    # volume mass within +-15 min of the expected open / close auctions
    near_open = prof[(prof.index >= open_min - 5) & (prof.index <= open_min + 15)].sum()
    near_close = prof[(prof.index >= close_min - 10) & (prof.index <= close_min + 10)].sum()
    top = prof.sort_values(ascending=False).head(6)
    return {
        "declared_tz": cfg.tz,
        "top_volume_minutes": [_hhmm(m) for m in top.index.tolist()],
        "volume_share_near_0900_open": round(float(near_open / prof.sum()), 4),
        "volume_share_near_1730_close": round(float(near_close / prof.sum()), 4),
        "consistent_with_berlin": bool(near_open / prof.sum() > 0.02
                                       and near_close / prof.sum() > 0.01),
    }


def _coverage_and_sessions(df: pd.DataFrame, cfg: Config) -> dict:
    s = sb.attach_sessions(df, cfg)
    cash = s[s["is_cash"]]
    per_day_cash = cash.groupby("session_date").size()
    first = s.groupby("session_date")["tod_min"].min()
    last = s.groupby("session_date")["tod_min"].max()
    return {
        "coverage_start": str(df["dt"].min()),
        "coverage_end": str(df["dt"].max()),
        "n_sessions": int(s["session_date"].nunique()),
        "session_first_bar_mode": _hhmm(first.mode().iloc[0]) if len(first) else None,
        "session_last_bar_mode": _hhmm(last.mode().iloc[0]) if len(last) else None,
        "cash_bars_per_day_median": float(per_day_cash.median()),
        "cash_bars_per_day_min": int(per_day_cash.min()),
        "cash_bars_per_day_max": int(per_day_cash.max()),
    }


def _missing_and_gaps(df: pd.DataFrame, cfg: Config, bar_minutes: int) -> dict:
    """Count missing bars within the liquid cash session grid (09:00-17:30)."""
    s = sb.attach_sessions(df, cfg)
    cash = s[s["is_cash"]]
    exp_per_day = ((cfg.cash_close.hour * 60 + cfg.cash_close.minute)
                   - cfg.cash_open_minutes()) // bar_minutes + 1
    present = cash.groupby("session_date").size()
    # only count on days that look like full sessions (avoid holidays/half-days)
    full_days = present[present >= 0.8 * exp_per_day]
    missing = (exp_per_day - full_days).clip(lower=0)
    # intraday gaps > bar size inside cash
    cg = cash.sort_values("dt")
    d = cg.groupby("session_date")["dt"].apply(
        lambda x: (x.diff().dropna() > pd.Timedelta(minutes=bar_minutes)).sum())
    return {
        "expected_cash_bars_per_full_day": int(exp_per_day),
        "full_session_days": int(len(full_days)),
        "missing_cash_bars_total_on_full_days": int(missing.sum()),
        "missing_cash_bars_pct_on_full_days": round(
            100 * missing.sum() / (exp_per_day * max(len(full_days), 1)), 3),
        "sessions_with_intraday_cash_gaps": int((d > 0).sum()),
    }


def audit_dataset(name: str, lr: LoadResult, cfg: Config, bar_minutes: int) -> dict:
    df = lr.df
    rep = {
        "dataset": name,
        "source_files": lr.source_files,
        "n_raw_rows": lr.n_raw,
        "n_rows_clean": len(df),
        "n_exact_duplicates_dropped": lr.n_exact_duplicates,
        "n_conflicting_duplicates": lr.n_conflicting_duplicates,
        "tz_localize_fallback_used": lr.localize_fallback_used,
        "bar_minutes": bar_minutes,
        "timezone_detection": detect_timezone(df, cfg),
        "coverage_sessions": _coverage_and_sessions(df, cfg),
        "missing_gaps": _missing_and_gaps(df, cfg, bar_minutes),
    }
    return rep


def audit_contracts(fdax: pd.DataFrame, cfg: Config) -> dict:
    g = fdax.groupby("contract_month")
    contracts = []
    for cm, gg in g:
        contracts.append({
            "contract_month": int(cm),
            "con_id": int(gg["con_id"].iloc[0]),
            "local_symbol": str(gg["local_symbol"].iloc[0]),
            "expiry": int(gg["expiry"].iloc[0]),
            "coverage": [str(gg["dt"].min()), str(gg["dt"].max())],
            "n_bars": int(len(gg)),
        })
    # overlap between contracts sharing a timestamp
    overlap = int((fdax.groupby("dt")["con_id"].nunique() > 1).sum())
    roll = sb.compute_roll_adjustment(fdax, cfg)
    return {
        "n_contracts": len(contracts),
        "contracts": contracts,
        "timestamps_with_multiple_contracts": overlap,
        "roll_method": roll.method,
        "roll_seams": roll.seams,
        "roll_verification": roll.verification,
    }


def run_full_audit(cfg: Config = DEFAULT_CONFIG, save: bool = True) -> dict:
    dax = load_dax_5m(cfg)
    fdax = load_fdax_1m(cfg)
    report = {
        "config": cfg.to_dict(),
        "dax_5m": audit_dataset("dax_5m", dax, cfg, bar_minutes=5),
        "fdax_1m": audit_dataset("fdax_1m", fdax, cfg, bar_minutes=1),
        "fdax_contracts": audit_contracts(fdax.df, cfg),
        "cross_dataset_gap": _cross_gap(dax.df, fdax.df),
        "warnings": [],
    }
    _collect_warnings(report)
    if save:
        cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.audit_report_json.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        cfg.paths.audit_report_txt.write_text(render_text(report), encoding="utf-8")
    return report


def _cross_gap(dax: pd.DataFrame, fdax: pd.DataFrame) -> dict:
    dax_end = dax["dt"].max()
    fdax_start = fdax["dt"].min()
    gap_days = (fdax_start.tz_convert("UTC") - dax_end.tz_convert("UTC")).days
    return {
        "dax_5m_ends": str(dax_end),
        "fdax_1m_starts": str(fdax_start),
        "gap_calendar_days": int(gap_days),
        "note": "No data between these dates; prior-ATH for early-FDAX sessions "
                "cannot see any highs made inside this window (reported, not filled).",
    }


def _collect_warnings(report: dict) -> None:
    w = report["warnings"]
    for key in ("dax_5m", "fdax_1m"):
        d = report[key]
        if not d["timezone_detection"]["consistent_with_berlin"]:
            w.append(f"{key}: volume profile not clearly consistent with Europe/Berlin cash session.")
        if d["n_conflicting_duplicates"]:
            w.append(f"{key}: {d['n_conflicting_duplicates']} conflicting duplicate timestamps kept-first.")
        if d["tz_localize_fallback_used"]:
            w.append(f"{key}: DST localization fell back to ambiguous=DST for some overnight bars.")
    if report["cross_dataset_gap"]["gap_calendar_days"] > 3:
        w.append(f"cross-dataset gap of {report['cross_dataset_gap']['gap_calendar_days']} "
                 f"days between DAX-5m end and FDAX-1m start.")
    rv = report["fdax_contracts"]["roll_verification"]
    if not rv.get("newest_contract_anchored_raw", False):
        w.append("roll back-adjustment: newest contract is NOT anchored raw (unexpected).")


def render_text(report: dict) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A("DAX ANALOG DAY EXPLORER — DATA AUDIT")
    A("=" * 78)
    for key in ("dax_5m", "fdax_1m"):
        d = report[key]
        A(f"\n[{d['dataset']}]  files={len(d['source_files'])}  bar={d['bar_minutes']}m")
        A(f"  rows: raw={d['n_raw_rows']}  clean={d['n_rows_clean']}  "
          f"exact_dupes_dropped={d['n_exact_duplicates_dropped']}  "
          f"conflicting={d['n_conflicting_duplicates']}")
        cs = d["coverage_sessions"]
        A(f"  coverage: {cs['coverage_start']}  ->  {cs['coverage_end']}  "
          f"({cs['n_sessions']} sessions)")
        A(f"  session hours (mode): {cs['session_first_bar_mode']} .. {cs['session_last_bar_mode']}  "
          f"| cash bars/day median={cs['cash_bars_per_day_median']}")
        tz = d["timezone_detection"]
        A(f"  timezone: declared={tz['declared_tz']}  consistent_with_berlin={tz['consistent_with_berlin']}"
          f"  top_vol_minutes={tz['top_volume_minutes']}")
        mg = d["missing_gaps"]
        A(f"  missing cash bars (full days): {mg['missing_cash_bars_total_on_full_days']} "
          f"({mg['missing_cash_bars_pct_on_full_days']}%)  "
          f"| sessions w/ intraday gaps={mg['sessions_with_intraday_cash_gaps']}")
    fc = report["fdax_contracts"]
    A(f"\n[FDAX contracts]  n={fc['n_contracts']}  simultaneous-overlap-bars="
      f"{fc['timestamps_with_multiple_contracts']}")
    for c in fc["contracts"]:
        A(f"  {c['contract_month']}  conId={c['con_id']}  exp={c['expiry']}  "
          f"{c['coverage'][0][:10]}..{c['coverage'][1][:10]}  bars={c['n_bars']}")
    A(f"\n[roll back-adjustment]  method={fc['roll_method']}")
    for s in fc["roll_seams"]:
        A(f"  {s['old']}->{s['new']}: gap={s['seam_gap_pct']:+.2f}%  "
          f"carry={s['carry_pct']:+.2f}%  real_move_est={s['real_move_pts_est']:+.0f}pts")
    rv = fc["roll_verification"]
    A(f"  verification: newest_raw={rv['newest_contract_anchored_raw']}  "
      f"preserves_returns={rv['preserves_pct_returns']}  "
      f"seam_residual_max={rv['seam_residual_pct_max']}%")
    A(f"  {rv['note']}")
    cg = report["cross_dataset_gap"]
    A(f"\n[cross-dataset gap] {cg['dax_5m_ends'][:10]} -> {cg['fdax_1m_starts'][:10]} "
      f"({cg['gap_calendar_days']} days)")
    if report["warnings"]:
        A("\n[WARNINGS]")
        for x in report["warnings"]:
            A(f"  ! {x}")
    A("=" * 78)
    return "\n".join(L)
