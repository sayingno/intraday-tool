"""Opening trend-continuation study and backtest.

The setup: the first ``window`` minutes trend one way and close near that
extreme (context label ``DRIVE_UP`` / ``DRIVE_DOWN``).  The trade takes that
direction at the decision bar and asks whether the move continues.

Everything here is leak-free by construction:

* the signal uses only bars strictly before the decision bar,
* the entry fill is the **open of the first bar after** the decision bar,
* the simulation then walks forward bar by bar,
* when a bar's range contains both the stop and the target, the **stop** is
  taken (the conservative assumption -- a 5-minute bar cannot tell us which came
  first).

Risk is expressed in **R**, where R is the distance from entry to the structural
stop, so 2003 and 2026 are comparable without any volatility indicator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb


@dataclass
class TradeRules:
    """One fully-specified strategy."""
    decision_min: int = 30            # signal complete at 09:30 (end of bar 6)
    stop_mode: str = "opening_range"  # {"opening_range", "half_range", "last_bar"}
    target_R: float | None = 2.0      # fixed target in R; None = no target
    breakeven_at_R: float | None = None   # move stop to entry once this is reached
    trail_bars: int | None = None     # trail behind the extreme of the last N bars
    time_exit_min: int | None = None  # flatten at this minute-since-open
    cost_points: float = 2.0          # round-trip spread + commission, in points

    def describe(self) -> list[str]:
        d = [f"enter at the open of the bar after {self.decision_min} min",
             f"stop = {self.stop_mode.replace('_', ' ')}"]
        d.append(f"target = {self.target_R:g}R" if self.target_R else "no fixed target")
        if self.breakeven_at_R:
            d.append(f"stop to breakeven at +{self.breakeven_at_R:g}R")
        if self.trail_bars:
            d.append(f"trail behind the last {self.trail_bars} bars")
        if self.time_exit_min:
            d.append(f"flat at {self.time_exit_min} min after the open")
        else:
            d.append("otherwise hold to the cash close")
        d.append(f"costs {self.cost_points:g} points round trip")
        return d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stop_price(obs: pd.DataFrame, side: int, mode: str, entry: float) -> float:
    hi, lo = float(obs["adj_high"].max()), float(obs["adj_low"].min())
    if mode == "opening_range":
        return lo if side > 0 else hi
    if mode == "half_range":
        mid = (hi + lo) / 2.0
        return mid if ((side > 0 and mid < entry) or (side < 0 and mid > entry)) else (lo if side > 0 else hi)
    if mode == "last_bar":
        last = obs.sort_values("mso").iloc[-1]
        return float(last["adj_low"]) if side > 0 else float(last["adj_high"])
    raise ValueError(mode)


def simulate_day(g: pd.DataFrame, side: int, rules: TradeRules) -> dict | None:
    """Walk one session forward from the decision bar.  Returns None if untradeable."""
    g = g.sort_values("mso")
    obs = g[g["mso"] < rules.decision_min]
    aft = g[g["mso"] >= rules.decision_min]
    if len(obs) < 3 or len(aft) < 2:
        return None
    entry = float(aft["adj_open"].iloc[0])
    stop = _stop_price(obs, side, rules.stop_mode, entry)
    R = abs(entry - stop)
    if R <= 0:
        return None

    target = entry + side * rules.target_R * R if rules.target_R else None
    be_done = False
    hi = aft["adj_high"].to_numpy(); lo = aft["adj_low"].to_numpy()
    cl = aft["adj_close"].to_numpy(); mso = aft["mso"].to_numpy()

    exit_price, exit_min, reason = float(cl[-1]), int(mso[-1]), "close"
    for i in range(len(aft)):
        # --- stop first: a 5m bar cannot prove the target came first ---
        hit_stop = (lo[i] <= stop) if side > 0 else (hi[i] >= stop)
        hit_tgt = target is not None and ((hi[i] >= target) if side > 0 else (lo[i] <= target))
        if hit_stop:
            exit_price, exit_min, reason = stop, int(mso[i]), ("breakeven" if be_done else "stop")
            break
        if hit_tgt:
            exit_price, exit_min, reason = target, int(mso[i]), "target"
            break
        if rules.time_exit_min is not None and mso[i] >= rules.time_exit_min:
            exit_price, exit_min, reason = float(cl[i]), int(mso[i]), "time"
            break
        # --- update the stop AFTER the bar has been evaluated ---
        run = (hi[i] - entry) if side > 0 else (entry - lo[i])
        if rules.breakeven_at_R and not be_done and run >= rules.breakeven_at_R * R:
            stop, be_done = entry, True
        if rules.trail_bars and i + 1 >= rules.trail_bars:
            w = slice(i + 1 - rules.trail_bars, i + 1)
            t = float(lo[w].min()) if side > 0 else float(hi[w].max())
            stop = max(stop, t) if side > 0 else min(stop, t)

    gross = side * (exit_price - entry)
    net = gross - rules.cost_points
    return {"entry": entry, "stop_init": _stop_price(obs, side, rules.stop_mode, entry),
            "R_points": R, "exit_price": exit_price, "exit_min": exit_min,
            "reason": reason, "gross_points": gross, "net_points": net,
            "R_multiple": net / R}


def session_groups(master: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> dict:
    """Cash-session bars keyed by date.  Build once, reuse across many rule sets --
    sessionizing the whole master per backtest dominates the runtime otherwise."""
    ms = sb.attach_sessions(master, cfg)
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0)]
    return dict(tuple(cash.groupby("session_date")))


def build_trades(master: pd.DataFrame | None, context_table: pd.DataFrame,
                 rules: TradeRules = TradeRules(), cfg: Config = DEFAULT_CONFIG,
                 labels=("DRIVE_UP", "DRIVE_DOWN"),
                 groups: dict | None = None) -> pd.DataFrame:
    if groups is None:
        groups = session_groups(master, cfg)
    lab = context_table.set_index("session_date")["opening"]
    rows = []
    for sd, g in groups.items():
        if sd not in lab.index or lab.loc[sd] not in labels:
            continue
        side = 1 if lab.loc[sd] == "DRIVE_UP" else -1
        r = simulate_day(g, side, rules)
        if r is None:
            continue
        r["session_date"] = sd
        r["side"] = side
        r["label"] = lab.loc[sd]
        rows.append(r)
    t = pd.DataFrame(rows)
    return t.sort_values("session_date").reset_index(drop=True) if len(t) else t


def performance(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {"n": 0}
    r = trades["R_multiple"]
    wins, losses = r[r > 0], r[r <= 0]
    gp, gl = wins.sum(), -losses.sum()
    eq = r.cumsum()
    dd = (eq.cummax() - eq).max()
    return {
        "n": len(r),
        "expectancy_R": round(float(r.mean()), 4),
        "median_R": round(float(r.median()), 4),
        "win_rate_%": round(100 * float((r > 0).mean()), 1),
        "avg_win_R": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_R": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else np.inf,
        "total_R": round(float(r.sum()), 1),
        "max_drawdown_R": round(float(dd), 1),
        # standard error of the mean -> is the expectancy distinguishable from zero?
        "se_R": round(float(r.std() / np.sqrt(len(r))), 4),
        "t_stat": round(float(r.mean() / (r.std() / np.sqrt(len(r)))), 2) if r.std() > 0 else 0.0,
    }


def split_performance(trades: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """In-sample vs out-of-sample, so a tuned rule has to prove itself twice."""
    s = pd.Timestamp(split_date)
    rows = []
    for name, sub in (("in-sample", trades[trades["session_date"] < s]),
                      ("out-of-sample", trades[trades["session_date"] >= s]),
                      ("all", trades)):
        p = performance(sub)
        p["period"] = name
        rows.append(p)
    cols = ["period", "n", "expectancy_R", "t_stat", "win_rate_%", "avg_win_R",
            "avg_loss_R", "profit_factor", "total_R", "max_drawdown_R"]
    return pd.DataFrame(rows)[cols]
