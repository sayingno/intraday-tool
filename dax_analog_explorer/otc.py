"""Open Trend Continuation -- the one executable DAX setup.

    open develops direction -> genuine follow-through -> controlled pullback
    -> continuation trigger in the same direction

This is a **sequence**, not a snapshot, so it is evaluated bar by bar rather than
labelled at a fixed cutoff like ``context.py`` and ``price_action.py`` do.

No fitted thresholds and no indicators anywhere in the decision path.  Every
condition is a count or a structural comparison between bars.  The only numbers
are scope choices: the bar interval, the trigger deadline, and how many entries a
session may produce.

Rule 3 ("not extended") deliberately has no test of its own.  A vertical climax
has not pulled back, so it cannot produce a trigger; and a leg that has over-run
leaves a wide structural stop while the next structural level sits close, so the
reported R:R fails on its own arithmetic.  Extension is filtered by geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb

# state machine
INVALID = "INVALID"          # A -- no direction
DEVELOPING = "DEVELOPING"    # B -- direction, no completed rotation yet
ACTIVE = "ACTIVE"            # C -- direction + rotation + trigger

TICK = 1.0                   # FDAX minimum price increment, in points


@dataclass
class OTCParams:
    """Scope only.  Nothing here is fitted to outcomes."""
    bar_minutes: int = 5
    leg_start_min: int = 0            # the leg is measured from the cash open
    trigger_deadline_min: int = 90    # 10:30 Berlin -- an *opening* continuation
    max_entries: int = 2
    tick: float = TICK
    cost_points: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# rule 1 + 2 -- direction and follow-through
# --------------------------------------------------------------------------- #
@dataclass
class Leg:
    side: int                 # +1 bull, -1 bear
    origin_px: float          # where the leg started (the cash open)
    origin_i: int
    extreme_px: float         # furthest point reached so far
    extreme_i: int
    hh_hl: int                # consecutive higher-high AND higher-low steps
    followthrough_bars: int   # distinct bars that extended beyond bar 1


def detect_leg(h: np.ndarray, l: np.ndarray, c: np.ndarray, o: np.ndarray,
               upto: int) -> Leg | None:
    """Rules 1 and 2, using bars 0..upto inclusive.

    Rule 1: >=2 consecutive higher highs AND higher lows (mirrored for a bear).
    Rule 2: >=2 distinct bars extend the move beyond bar 1's extreme -- which
    rejects "one big bar then nothing" without ever measuring a bar's size.
    """
    n = upto + 1
    if n < 3:
        return None
    for side in (1, -1):
        steps = 0
        best = 0
        for i in range(1, n):
            if side > 0:
                up = h[i] > h[i - 1] and l[i] > l[i - 1]
            else:
                up = h[i] < h[i - 1] and l[i] < l[i - 1]
            steps = steps + 1 if up else 0
            best = max(best, steps)
        if best < 2:
            continue
        # follow-through: distinct bars beyond bar 1's extreme
        if side > 0:
            ft = int(np.sum(h[1:n] > h[0]))
            ext_i = int(np.argmax(h[:n])); ext = float(h[ext_i])
        else:
            ft = int(np.sum(l[1:n] < l[0]))
            ext_i = int(np.argmin(l[:n])); ext = float(l[ext_i])
        if ft < 2:
            continue
        origin = float(l[0]) if side > 0 else float(h[0])
        return Leg(side=side, origin_px=origin, origin_i=0,
                   extreme_px=ext, extreme_i=ext_i, hh_hl=best,
                   followthrough_bars=ft)
    return None


# --------------------------------------------------------------------------- #
# rule 5 -- Brooks bar counting: H1/H2 (buy) and L1/L2 (sell)
# --------------------------------------------------------------------------- #
@dataclass
class CountSignal:
    kind: str            # "H" (buy) or "L" (sell)
    count: int           # 1 = H1/L1, 2 = H2/L2, ...
    signal_i: int        # the signal bar
    entry_px: float      # stop order 1 tick beyond the signal bar
    stop_px: float       # 1 tick beyond the structure
    pullback_extreme: float

    @property
    def label(self) -> str:
        return f"{self.kind}{self.count}"


def bar_count(h: np.ndarray, l: np.ndarray, leg: Leg, upto: int,
              tick: float = TICK) -> list[CountSignal]:
    """The canonical count, run inside the correction against ``leg``.

    Bull: the pullback opens on a bar with a LOWER HIGH; the first later bar whose
    HIGH exceeds the prior bar's high is H1; after another leg down, the next such
    bar is H2.  The count RESETS when price makes a new leg extreme.
    Bear is the exact mirror on lows.
    """
    kind = "H" if leg.side > 0 else "L"
    out: list[CountSignal] = []
    count = 0
    leg_against = False          # is a counter-leg under way?
    # The extreme must be tracked CAUSALLY, as the running max/min of the bars
    # walked so far.  Seeding it from leg.extreme_px is wrong: that value is
    # computed over the whole window and therefore already contains any later new
    # high, so the reset would never fire and every post-reset H1 would be
    # mislabelled H2.
    extreme = float(h[0]) if leg.side > 0 else float(l[0])
    pb_ext = None                # furthest point of the current correction

    start = max(1, leg.origin_i + 1)
    for i in range(start, upto + 1):
        if leg.side > 0:
            # trend resumed -> the count resets
            if h[i] > extreme:
                extreme = float(h[i]); count = 0; leg_against = False; pb_ext = None
                continue
            if h[i] < h[i - 1]:
                leg_against = True
                pb_ext = float(l[i]) if pb_ext is None else min(pb_ext, float(l[i]))
            elif leg_against and h[i] > h[i - 1]:
                count += 1
                struct = min(float(l[i]), pb_ext if pb_ext is not None else float(l[i]))
                out.append(CountSignal(kind, count, i,
                                       float(h[i]) + tick, struct - tick,
                                       pb_ext if pb_ext is not None else float(l[i])))
                leg_against = False
        else:
            if l[i] < extreme:
                extreme = float(l[i]); count = 0; leg_against = False; pb_ext = None
                continue
            if l[i] > l[i - 1]:
                leg_against = True
                pb_ext = float(h[i]) if pb_ext is None else max(pb_ext, float(h[i]))
            elif leg_against and l[i] < l[i - 1]:
                count += 1
                struct = max(float(h[i]), pb_ext if pb_ext is not None else float(h[i]))
                out.append(CountSignal(kind, count, i,
                                       float(l[i]) - tick, struct + tick,
                                       pb_ext if pb_ext is not None else float(h[i])))
                leg_against = False
    return out


# --------------------------------------------------------------------------- #
# rule 4 -- did the correction keep the structure?
# --------------------------------------------------------------------------- #
def origin_held(h: np.ndarray, l: np.ndarray, leg: Leg, upto: int) -> bool:
    """The pullback must not trade beyond where the leg began."""
    lo = leg.origin_i + 1
    if lo > upto:
        return True
    if leg.side > 0:
        return float(np.min(l[lo:upto + 1])) >= leg.origin_px
    return float(np.max(h[lo:upto + 1])) <= leg.origin_px


# --------------------------------------------------------------------------- #
# rule 7 -- the next structural destination
# --------------------------------------------------------------------------- #
def next_structural_target(entry: float, side: int, levels: dict[str, float],
                           session_extreme: float | None = None) -> tuple[str, float] | None:
    """Nearest untouched level beyond the entry, in the trade direction."""
    cands = []
    for name, v in levels.items():
        if v is None or pd.isna(v):
            continue
        if (side > 0 and v > entry) or (side < 0 and v < entry):
            cands.append((abs(v - entry), name, float(v)))
    if session_extreme is not None and not pd.isna(session_extreme):
        if (side > 0 and session_extreme > entry) or (side < 0 and session_extreme < entry):
            cands.append((abs(session_extreme - entry), "SESSION_EXTREME",
                          float(session_extreme)))
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2]


# --------------------------------------------------------------------------- #
# the session walk
# --------------------------------------------------------------------------- #
@dataclass
class OTCSignal:
    label: str                # H1 / H2 / L1 ...
    side: int
    signal_mso: int
    entry_px: float
    stop_px: float
    target_name: str | None
    target_px: float | None
    rr: float | None
    taken: bool               # False once max_entries is exhausted
    pullback_px: float | None = None   # the correction's extreme, for charting
    # the leg AS IT STOOD when this signal fired.  A session can build a bear leg,
    # fail, and then build a bull one, so the result's final leg is not necessarily
    # the leg that produced any given signal -- carrying it per signal keeps the
    # reporting and the charts honest.
    leg_origin_px: float | None = None
    leg_extreme_px: float | None = None
    leg_extreme_i: int | None = None
    reason: str = ""


@dataclass
class OTCResult:
    state: str
    leg: Leg | None
    signals: list[OTCSignal] = field(default_factory=list)
    state_trace: list[tuple[int, str]] = field(default_factory=list)


def evaluate_session(bars: pd.DataFrame, levels: dict[str, float],
                     p: OTCParams = OTCParams()) -> OTCResult:
    """Walk the session bar by bar and emit every OTC signal up to the deadline."""
    b = bars.sort_values("mso")
    h = b["adj_high"].to_numpy(); l = b["adj_low"].to_numpy()
    c = b["adj_close"].to_numpy(); o = b["adj_open"].to_numpy()
    mso = b["mso"].to_numpy()
    n = len(b)
    res = OTCResult(state=INVALID, leg=None)
    if n < 4:
        return res

    seen: set[int] = set()
    taken = 0
    for i in range(2, n):
        if mso[i] > p.trigger_deadline_min:
            break
        leg = detect_leg(h, l, c, o, i)
        if leg is None:
            res.state_trace.append((int(mso[i]), INVALID))
            continue
        res.leg = leg
        if not origin_held(h, l, leg, i):
            res.state = INVALID
            res.state_trace.append((int(mso[i]), INVALID))
            continue

        sigs = bar_count(h, l, leg, i, p.tick)
        new = [s for s in sigs if s.signal_i not in seen and s.signal_i == i]
        if not new:
            res.state = DEVELOPING
            res.state_trace.append((int(mso[i]), DEVELOPING))
            continue

        for s in new:
            seen.add(s.signal_i)
            sess_ext = float(np.max(h[:i + 1])) if leg.side > 0 else float(np.min(l[:i + 1]))
            tgt = next_structural_target(s.entry_px, leg.side, levels, sess_ext)
            risk = abs(s.entry_px - s.stop_px)
            rr = (abs(tgt[1] - s.entry_px) / risk) if (tgt and risk > 0) else None
            ok = taken < p.max_entries
            res.signals.append(OTCSignal(
                label=s.label, side=leg.side, signal_mso=int(mso[i]),
                entry_px=s.entry_px, stop_px=s.stop_px,
                target_name=tgt[0] if tgt else None,
                target_px=tgt[1] if tgt else None,
                rr=round(rr, 2) if rr else None, taken=ok,
                pullback_px=s.pullback_extreme,
                leg_origin_px=leg.origin_px, leg_extreme_px=leg.extreme_px,
                leg_extreme_i=leg.extreme_i,
                reason="" if ok else "max_entries reached"))
            if ok:
                taken += 1
        res.state = ACTIVE
        res.state_trace.append((int(mso[i]), ACTIVE))
    return res


# --------------------------------------------------------------------------- #
# the live sentence
# --------------------------------------------------------------------------- #
def live_status(res: OTCResult, p: OTCParams = OTCParams()) -> str:
    if res.leg is None:
        return ("**OTC status: INVALID.**\nNo established direction from the open.\n"
                "**Decision: PASS** — DAX gives nothing today.")
    d = "bull" if res.leg.side > 0 else "bear"
    lines = [f"**OTC status: {res.state}.**", f"Direction: {d}.",
             f"Follow-through: {res.leg.followthrough_bars} bars beyond bar 1 "
             f"({'sufficient' if res.leg.followthrough_bars >= 2 else 'insufficient'})."]
    live = [s for s in res.signals if s.taken]
    if res.state == ACTIVE and live:
        s = live[-1]
        lines += [f"Current state: {s.label} continuation trigger.",
                  f"Trigger: {'above' if s.side > 0 else 'below'} {s.entry_px:.0f}.",
                  f"Stop: {'below' if s.side > 0 else 'above'} {s.stop_px:.0f}.",
                  f"T1: {s.target_px:.0f} ({s.target_name})." if s.target_px
                  else "T1: no untouched structural level ahead.",
                  f"R:R: {s.rr}." if s.rr else "R:R: not computable.",
                  "**Decision: TAKE**" if (s.rr or 0) >= 1 else
                  "**Decision: TAKE — but R:R below 1, your call.**"]
    elif res.state == DEVELOPING:
        lines += ["Current state: direction established, no completed rotation yet.",
                  "**Decision: WAIT** — do not enter an extended move."]
    else:
        lines += ["Current state: structure broken (leg origin violated).",
                  "**Decision: PASS**"]
    return "\n".join(lines)
