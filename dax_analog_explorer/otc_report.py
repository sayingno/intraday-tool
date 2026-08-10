"""Open Trend Continuation -- scan, backtest and chart pack.

    python -m dax_analog_explorer.otc_report --profile   # how many days reach A/B/C
    python -m dax_analog_explorer.otc_report             # backtest, split by bar count
    python -m dax_analog_explorer.otc_report --pdf otc.pdf
    python -m dax_analog_explorer.otc_report --date 2025-05-12   # the live sentence

The entry is event driven: a stop order 1 tick beyond the counted signal bar,
filled only if a later bar actually trades through it.  A signal that never
trades is NOT_TRIGGERED -- it is not a loss, and counting it as one would inflate
every loss statistic with trades that were never taken.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import Config
from . import otc
from . import continuation as co


def simulate_signal(bars: pd.DataFrame, sig: otc.OTCSignal, p: otc.OTCParams) -> dict:
    """Walk forward from the signal bar.  Stop wins on an ambiguous bar."""
    b = bars.sort_values("mso")
    fwd = b[b["mso"] > sig.signal_mso]
    if len(fwd) == 0:
        return {"status": "NOT_TRIGGERED"}
    hi = fwd["adj_high"].to_numpy(); lo = fwd["adj_low"].to_numpy()
    cl = fwd["adj_close"].to_numpy(); mso = fwd["mso"].to_numpy()
    side = sig.side
    risk = abs(sig.entry_px - sig.stop_px)
    if risk <= 0:
        return {"status": "NOT_TRIGGERED"}

    entered = False
    for i in range(len(fwd)):
        if not entered:
            hit = (hi[i] >= sig.entry_px) if side > 0 else (lo[i] <= sig.entry_px)
            if not hit:
                # the structure the signal rests on can die before the entry ever
                # fills: once price trades through the level where the stop would
                # sit, the pullback has broken and no trader leaves that order
                # working.  Filling it later, with a stop that has already been
                # violated, is a trade nobody took.
                dead = (lo[i] <= sig.stop_px) if side > 0 else (hi[i] >= sig.stop_px)
                if dead:
                    return {"status": "CANCELLED"}
                continue
            entered = True
            # a bar that triggers can also stop within the same bar -> stop wins
            stopped = (lo[i] <= sig.stop_px) if side > 0 else (hi[i] >= sig.stop_px)
            if stopped:
                return _res(sig, sig.stop_px, int(mso[i]), "stop", risk, p)
            continue
        stopped = (lo[i] <= sig.stop_px) if side > 0 else (hi[i] >= sig.stop_px)
        if stopped:
            return _res(sig, sig.stop_px, int(mso[i]), "stop", risk, p)
        if sig.target_px is not None:
            hitt = (hi[i] >= sig.target_px) if side > 0 else (lo[i] <= sig.target_px)
            if hitt:
                return _res(sig, sig.target_px, int(mso[i]), "target", risk, p)
    if not entered:
        return {"status": "NOT_TRIGGERED"}
    return _res(sig, float(cl[-1]), int(mso[-1]), "close", risk, p)


def _res(sig, exit_px, exit_mso, reason, risk, p):
    gross = sig.side * (exit_px - sig.entry_px)
    net = gross - p.cost_points
    return {"status": "TRADED", "exit_px": exit_px, "exit_mso": exit_mso,
            "reason": reason, "net_points": net, "R_multiple": net / risk,
            "risk_points": risk}


def build_otc_trades(groups: dict, daily: pd.DataFrame,
                     p: otc.OTCParams = otc.OTCParams()) -> pd.DataFrame:
    di = daily.set_index("session_date")
    rows = []
    for sd, g in groups.items():
        if sd not in di.index:
            continue
        r = di.loc[sd]
        res = otc.evaluate_session(g, _levels(r), p)
        for s in res.signals:
            row = {"session_date": sd, "label": s.label, "count": int(s.label[1:]),
                   "kind": s.label[0], "side": s.side, "signal_mso": s.signal_mso,
                   "entry_px": s.entry_px, "stop_px": s.stop_px,
                   "target_name": s.target_name, "target_px": s.target_px,
                   "rr": s.rr, "taken": s.taken, "state": res.reached}
            row.update(simulate_signal(g, s, p) if s.taken else {"status": "NOT_TAKEN"})
            rows.append(row)
    t = pd.DataFrame(rows)
    return t.sort_values(["session_date", "signal_mso"]).reset_index(drop=True) if len(t) else t


def _levels(row: pd.Series) -> dict:
    return {k: row.get(v) for k, v in
            (("PDH", "previous_day_high"), ("PDC", "previous_close"),
             ("PDL", "previous_day_low"), ("ONH", "overnight_high_adj"),
             ("ONL", "overnight_low_adj"))}


def session_mark(bars: pd.DataFrame, res: otc.OTCResult,
                 sims: list[dict] | None = None) -> dict:
    """What a chart should draw for one session, from the engine's own decision.

    Nothing here is recomputed from prices: the leg extreme is the bar the engine
    called the extreme, and each signal's trigger/stop/target are the values it
    traded.  If the drawing and the numbers ever disagree, the bug is visible.

    ``sims`` is the outcome of each signal in ``res.signals`` order, when it is
    already known; without it the anatomy is drawn but no exits are.
    """
    mso = bars.sort_values("mso")["mso"].to_numpy()
    sigs = []
    for k, s in enumerate(res.signals):
        row = {"label": s.label, "signal_mso": s.signal_mso, "entry_px": s.entry_px,
               "stop_px": s.stop_px, "target_px": s.target_px,
               "target_name": s.target_name, "pullback_px": s.pullback_px,
               "taken": s.taken, "side": s.side,
               "leg_origin_px": s.leg_origin_px, "leg_extreme_px": s.leg_extreme_px,
               "leg_extreme_mso": (int(mso[s.leg_extreme_i])
                                   if s.leg_extreme_i is not None else None)}
        sim = sims[k] if sims is not None and k < len(sims) else None
        if sim:
            row["status"] = sim.get("status")
            if sim.get("status") == "TRADED":
                row.update(exit_mso=sim.get("exit_mso"), exit_px=sim.get("exit_px"),
                           reason=sim.get("reason"), R_multiple=sim.get("R_multiple"))
        sigs.append(row)
    return {"side": sigs[-1]["side"] if sigs else (res.leg.side if res.leg else 1),
            "state": res.state, "reached": res.reached, "signals": sigs}


def chart_marks(groups: dict, daily: pd.DataFrame, dates, trades: pd.DataFrame,
                p: otc.OTCParams) -> dict:
    """``session_mark`` for each date, with outcomes joined from the trade table."""
    di = daily.set_index("session_date")
    ti = trades.set_index(["session_date", "signal_mso"]) if len(trades) else None
    marks = {}
    for d in dates:
        d = pd.Timestamp(d).normalize()
        if d not in groups or d not in di.index:
            continue
        g = groups[d].sort_values("mso")
        res = otc.evaluate_session(g, _levels(di.loc[d]), p)
        if res.leg is None:
            continue
        sims = []
        for s in res.signals:
            if ti is not None and (d, s.signal_mso) in ti.index:
                t = ti.loc[(d, s.signal_mso)]
                t = t.iloc[0] if isinstance(t, pd.DataFrame) else t
                sims.append(t.to_dict())
            else:
                sims.append(None)
        sigs = session_mark(g, res, sims)["signals"]
        marks[d] = {"side": sigs[-1]["side"] if sigs else res.leg.side,
                    "state": res.state, "signals": sigs}
    return marks


def _perf_line(name, sub):
    tr = sub[sub.status == "TRADED"]
    if len(tr) == 0:
        return f"  {name:<10}{len(sub):>6} signals   no fills"
    p = co.performance(tr.rename(columns={"R_multiple": "R_multiple"}))
    return (f"  {name:<10}{len(sub):>6} signals {len(tr):>6} filled   "
            f"exp {p['expectancy_R']:>+7.3f}R  t {p['t_stat']:>6.2f}  "
            f"win {p['win_rate_%']:>5.1f}%  pf {p['profit_factor']:>5.2f}")


def main():
    ap = argparse.ArgumentParser(description="Open Trend Continuation study")
    ap.add_argument("--deadline", type=int, default=90, help="minutes after the open")
    ap.add_argument("--max-entries", type=int, default=2, dest="max_entries")
    ap.add_argument("--cost", type=float, default=2.0)
    ap.add_argument("--split", default="2017-01-01")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--date", default=None, help="print the live sentence for one date")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--pdf-max", type=int, default=24, dest="pdf_max")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    cfg = Config()
    p = otc.OTCParams(trigger_deadline_min=a.deadline, max_entries=a.max_entries,
                      cost_points=a.cost)
    daily = pd.read_parquet(cfg.paths.daily_features)
    master = pd.read_parquet(cfg.paths.master_5m)
    groups = co.session_groups(master, cfg)

    if a.date:
        sd = pd.Timestamp(a.date).normalize()
        di = daily.set_index("session_date")
        if sd not in di.index or sd not in groups:
            lo, hi = daily.session_date.min().date(), daily.session_date.max().date()
            print(f"no session on {sd.date()} — the data covers {lo} .. {hi}.")
            print("  (a later date needs its bars appended to a FDAX_1min_*.csv "
                  "and 'python -m dax_analog_explorer.preprocess' re-run)")
            raise SystemExit(1)
        r = di.loc[sd]
        print(otc.live_status(otc.evaluate_session(groups[sd], _levels(r), p), p))
        return

    if a.profile:
        counts = {otc.INVALID: 0, otc.DEVELOPING: 0, otc.ACTIVE: 0}
        di = daily.set_index("session_date")
        for sd, g in groups.items():
            if sd not in di.index:
                continue
            # the FURTHEST state reached, not the state on the final bar
            counts[otc.evaluate_session(g, _levels(di.loc[sd]), p).reached] += 1
        tot = sum(counts.values())
        print(f"Furthest OTC state reached by the {a.deadline}-minute deadline, "
              f"{tot} sessions\n")
        for k in (otc.ACTIVE, otc.DEVELOPING, otc.INVALID):
            print(f"  {k:<12}{counts[k]:>6}  ({100*counts[k]/tot:>5.1f}%)")
        return

    trades = build_otc_trades(groups, daily, p)
    if len(trades) == 0:
        print("no OTC signals")
        return

    print("=" * 92)
    print(f"OTC SIGNALS — deadline {a.deadline} min, max {a.max_entries} entries/session")
    print("=" * 92)
    print(f"  sessions producing a signal : {trades.session_date.nunique()}")
    print(f"  signals total               : {len(trades)}")
    for k, v in trades.status.value_counts().items():
        print(f"    {k:<16}{v:>6}")

    tr = trades[trades.status == "TRADED"]
    print("\n" + "=" * 92)
    print(f"PERFORMANCE — {len(tr)} filled trades")
    print("=" * 92)
    print(co.split_performance(tr, a.split).to_string(index=False))

    print("\nBY BAR COUNT  (the mandate's premise: the second attempt is the reliable one)")
    for lab in sorted(trades.label.unique()):
        print(_perf_line(lab, trades[trades.label == lab]))
    print("\n  H1+H2 vs H3+")
    print(_perf_line("H1/L1", trades[trades["count"] == 1]))
    print(_perf_line("H2/L2", trades[trades["count"] == 2]))
    print(_perf_line("H3+/L3+", trades[trades["count"] >= 3]))

    if len(tr):
        print("\nexit reasons:")
        for k, v in tr.reason.value_counts().items():
            print(f"   {k:<10}{v:>5}  mean {tr[tr.reason==k].R_multiple.mean():+.2f}R")
        print(f"\nR:R at signal — median {trades.rr.median():.2f}, "
              f"share >= 1: {100*(trades.rr >= 1).mean():.0f}%")
    if len(tr) < 30:
        print(f"\n  ! only {len(tr)} filled trades — treat every rate as indicative.")

    if a.csv:
        trades.to_csv(a.csv, index=False); print(f"\nwrote {a.csv}")
    if a.pdf:
        from . import chart_pdf as cp
        dates = list(dict.fromkeys(tr.session_date.tolist()))[-a.pdf_max:]
        marks = chart_marks(groups, daily, dates, trades, p)
        cp.otc_pdf(a.pdf, dates, master, daily, marks, deadline=a.deadline,
                   lookback=1, cfg=cfg,
                   subtitle=f"H1/H2 and L1/L2 continuation triggers · trigger deadline "
                            f"{a.deadline} min after the open · {len(dates)} sessions")
        print(f"wrote {a.pdf} ({len(dates)} sessions)")


if __name__ == "__main__":
    main()
