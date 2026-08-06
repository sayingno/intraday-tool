"""Pure price-action setup filtering and conditional-probability reporting.

Nothing in this module divides by an indicator.  Every "pressure" test is a
relationship between bar opens/highs/lows/closes, self-scaled by the bars' own
ranges -- the way the structure is read off a chart by eye.

Two objects:

``PriceActionSpec``
    The full parameter set for a setup: market context (ATH proximity, gap band,
    open vs PDH, location in the overnight range) plus opening-pressure tests
    measured over the first N minutes.

``conditional_report``
    Answers the trader's question at the decision bar: *given a level has NOT
    been tested yet, what are the odds it trades there later?*  Levels already
    reached before the decision bar are excluded from their own denominator --
    that is what makes the probability conditional rather than unconditional.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb


# --------------------------------------------------------------------------- #
# parameter set
# --------------------------------------------------------------------------- #
@dataclass
class PriceActionSpec:
    """Every parameter needed to reproduce a setup.  All optional -> None/False
    disables that condition, so a spec is also a readable description."""

    # ---- group 1: market context (known at the open) ----
    ath_tolerance_pct: float | None = 0.50      # |open - prior_ath| / prior_ath * 100 <=
    require_new_ath_at_open: bool = False       # open > prior_ath
    require_open_above_pdh: bool = True
    require_open_below_pdl: bool = False
    gap_min_pct: float | None = 0.30            # cash_open vs previous 17:30 close
    gap_max_pct: float | None = 1.20
    on_location_min: float | None = 0.70        # (open-ON_low)/(ON_high-ON_low)
    on_location_max: float | None = None
    vol_regimes: tuple[str, ...] | None = None

    # ---- group 2: opening pressure, pure bar geometry ----
    pressure_window_min: int = 15               # bars 1-3 on a 5m chart
    direction: str = "down"                     # "down" = selling pressure
    require_close_beyond_open: bool = True      # (A) close < open   (down)
    close_location_max: float | None = 0.33     # (B) close in bottom third of window range
    require_close_beyond_prior_bar: bool = False  # (E) a close below the prior bar's low
    require_monotonic_highs: bool = False       # (C) h1>h2>h3
    min_directional_bars: int | None = None     # (D) count(close<open) >=
    require_beyond_open_excursion: bool = False # (F) traded above the open then closed below

    # ---- group 3: decision point ----
    decision_bar: int = 6                       # bar 6 -> 09:30 on a 5m chart
    bar_minutes: int = 5

    # ---- group 4: conditional gate ----
    require_untested: tuple[str, ...] = ("PDH", "PDC", "PDL")

    # ---- group 5: outcome measurement ----
    morning_end_min: int = 180                  # 12:00 Berlin

    # ------------------------------------------------------------------ #
    @property
    def decision_min(self) -> int:
        return self.decision_bar * self.bar_minutes

    def describe(self) -> list[str]:
        """Human-readable condition chain, in evaluation order."""
        d = []
        if self.ath_tolerance_pct is not None:
            d.append(f"|open − prior ATH| ≤ {self.ath_tolerance_pct:g}%")
        if self.require_new_ath_at_open:
            d.append("open above prior ATH (new ATH)")
        if self.require_open_above_pdh:
            d.append("open > PDH")
        if self.require_open_below_pdl:
            d.append("open < PDL")
        if self.gap_min_pct is not None:
            d.append(f"gap ≥ {self.gap_min_pct:+g}%")
        if self.gap_max_pct is not None:
            d.append(f"gap ≤ {self.gap_max_pct:+g}%")
        if self.on_location_min is not None:
            d.append(f"open ≥ {self.on_location_min:.0%} of the overnight range")
        if self.on_location_max is not None:
            d.append(f"open ≤ {self.on_location_max:.0%} of the overnight range")
        if self.vol_regimes:
            d.append(f"volatility regime in {list(self.vol_regimes)}")
        w, side = self.pressure_window_min, ("below" if self.direction == "down" else "above")
        if self.require_close_beyond_open:
            d.append(f"close at +{w}m {side} the session open")
        if self.close_location_max is not None:
            frac = self.close_location_max
            d.append(f"close in the {'bottom' if self.direction=='down' else 'top'} "
                     f"{frac:.0%} of the first {w}m range")
        if self.require_close_beyond_prior_bar:
            d.append(f"a bar closed {side} the prior bar's {'low' if self.direction=='down' else 'high'}")
        if self.require_monotonic_highs:
            d.append("lower highs" if self.direction == "down" else "higher lows")
        if self.min_directional_bars:
            kind = "bear" if self.direction == "down" else "bull"
            d.append(f"≥{self.min_directional_bars} {kind} bars in the first {w}m")
        if self.require_beyond_open_excursion:
            d.append(f"traded beyond the open, then closed back {side} it")
        return d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# the worked example from the analysis
ATH_TRAP_SPEC = PriceActionSpec()


# --------------------------------------------------------------------------- #
# per-session opening geometry
# --------------------------------------------------------------------------- #
def opening_geometry(bars: pd.DataFrame, cash_open: float, window_min: int) -> dict:
    """Bar-geometry description of the first ``window_min`` minutes.

    All ratios are scaled by the window's own range -- never by an indicator.
    """
    w = bars[(bars["mso"] >= 0) & (bars["mso"] < window_min)].sort_values("mso")
    n = len(w)
    if n == 0:
        return {}
    O = w["adj_open"].to_numpy(); H = w["adj_high"].to_numpy()
    L = w["adj_low"].to_numpy();  C = w["adj_close"].to_numpy()
    hi, lo, c = float(H.max()), float(L.min()), float(C[-1])
    rng = hi - lo
    return {
        "n_bars": n,
        "win_high": hi, "win_low": lo, "win_close": c,
        "close_below_open": bool(c < cash_open),
        "close_above_open": bool(c > cash_open),
        # where the close sits inside the window's own range: 0 = on the low
        "close_location": (c - lo) / rng if rng > 0 else np.nan,
        "n_bear_bars": int((C < O).sum()),
        "n_bull_bars": int((C > O).sum()),
        "monotonic_lower_highs": bool(n >= 3 and np.all(np.diff(H) < 0)),
        "monotonic_higher_lows": bool(n >= 3 and np.all(np.diff(L) > 0)),
        "close_below_prior_low": bool(n >= 2 and np.any(C[1:] < L[:-1])),
        "close_above_prior_high": bool(n >= 2 and np.any(C[1:] > H[:-1])),
        "poked_above_open": bool(hi > cash_open),
        "poked_below_open": bool(lo < cash_open),
        "spike_up_reject": bool(hi > cash_open and c < cash_open),
        "spike_down_reject": bool(lo < cash_open and c > cash_open),
        "open_is_high": bool(hi <= cash_open + 1e-9),
        "open_is_low": bool(lo >= cash_open - 1e-9),
    }


def _passes_pressure(g: dict, spec: PriceActionSpec) -> bool:
    if not g:
        return False
    down = spec.direction == "down"
    if spec.require_close_beyond_open:
        if not (g["close_below_open"] if down else g["close_above_open"]):
            return False
    if spec.close_location_max is not None:
        loc = g["close_location"]
        if pd.isna(loc):
            return False
        # for an up-setup the mirror test is close_location >= 1 - max
        if down and loc > spec.close_location_max:
            return False
        if not down and loc < (1.0 - spec.close_location_max):
            return False
    if spec.require_close_beyond_prior_bar:
        if not (g["close_below_prior_low"] if down else g["close_above_prior_high"]):
            return False
    if spec.require_monotonic_highs:
        if not (g["monotonic_lower_highs"] if down else g["monotonic_higher_lows"]):
            return False
    if spec.min_directional_bars is not None:
        k = g["n_bear_bars"] if down else g["n_bull_bars"]
        if k < spec.min_directional_bars:
            return False
    if spec.require_beyond_open_excursion:
        if not (g["spike_up_reject"] if down else g["spike_down_reject"]):
            return False
    return True


def _passes_context(r: pd.Series, spec: PriceActionSpec) -> bool:
    if spec.ath_tolerance_pct is not None:
        v = r.get("distance_open_to_ath_pct")
        if pd.isna(v) or abs(v) > spec.ath_tolerance_pct:
            return False
    if spec.require_new_ath_at_open and not bool(r.get("ath_at_open", False)):
        return False
    if spec.require_open_above_pdh and not bool(r.get("open_above_pdh", False)):
        return False
    if spec.require_open_below_pdl and not bool(r.get("open_below_pdl", False)):
        return False
    gap = r.get("gap_pct")
    if spec.gap_min_pct is not None and (pd.isna(gap) or gap < spec.gap_min_pct):
        return False
    if spec.gap_max_pct is not None and (pd.isna(gap) or gap > spec.gap_max_pct):
        return False
    loc = r.get("open_location_in_overnight_range")
    if spec.on_location_min is not None and (pd.isna(loc) or loc < spec.on_location_min):
        return False
    if spec.on_location_max is not None and (pd.isna(loc) or loc > spec.on_location_max):
        return False
    if spec.vol_regimes and r.get("daily_volatility_regime") not in spec.vol_regimes:
        return False
    return True


# --------------------------------------------------------------------------- #
# level bookkeeping
# --------------------------------------------------------------------------- #
#   name -> (column in daily_features, direction price must travel to reach it)
LEVELS = {
    "PDH": ("previous_day_high", "down"),
    "PDC": ("previous_close", "down"),
    "PDL": ("previous_day_low", "down"),
    "ONH": ("overnight_high_adj", "up"),
    "ONL": ("overnight_low_adj", "down"),
}


def _reached(frame: pd.DataFrame, level: float, direction: str) -> bool:
    if pd.isna(level) or len(frame) == 0:
        return False
    if direction == "down":
        return bool((frame["adj_low"] <= level).any())
    return bool((frame["adj_high"] >= level).any())


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def build_session_table(master: pd.DataFrame, daily: pd.DataFrame,
                        spec: PriceActionSpec = ATH_TRAP_SPEC,
                        cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """One row per session: context columns, opening geometry, level state and
    outcomes.  Computed ONCE; specs are then applied as boolean masks, so a
    funnel costs nothing extra.

    Only the timing parameters (pressure window, decision bar, morning end)
    affect this table -- the threshold parameters do not.
    """
    ms = sb.attach_sessions(master, cfg)
    cash = ms[ms["is_cash"] & (ms["mso"] >= 0)]
    groups = dict(tuple(cash.groupby("session_date")))
    di = daily.set_index("session_date")
    dm = spec.decision_min
    up = "up" if spec.direction == "down" else "down"

    rows = []
    for sd, g in groups.items():
        if sd not in di.index:
            continue
        r = di.loc[sd]
        o = r.get("cash_open_adj")
        if pd.isna(o):
            continue
        g = g.sort_values("mso")
        obs = g[g["mso"] < dm]
        aft = g[g["mso"] >= dm]
        if len(obs) < spec.decision_bar or len(aft) == 0:
            continue
        geo = opening_geometry(g, o, spec.pressure_window_min)
        if not geo:
            continue
        morn = g[(g["mso"] >= dm) & (g["mso"] <= spec.morning_end_min)]
        cut_px = float(obs["adj_close"].iloc[-1])

        rec = {"session_date": sd,
               "gap_pct": r.get("gap_pct"),
               "distance_open_to_ath_pct": r.get("distance_open_to_ath_pct"),
               "ath_at_open": bool(r.get("ath_at_open", False)),
               "open_above_pdh": bool(r.get("open_above_pdh", False)),
               "open_below_pdl": bool(r.get("open_below_pdl", False)),
               "open_location_in_overnight_range": r.get("open_location_in_overnight_range"),
               "daily_volatility_regime": r.get("daily_volatility_regime"),
               "cash_open": o, "decision_price": cut_px}
        rec.update({f"pa_{k}": v for k, v in geo.items()})
        for name, (col, direction) in LEVELS.items():
            lvl = r.get(col)
            rec[f"level_{name}"] = lvl
            rec[f"pre_{name}"] = _reached(obs, lvl, direction)
            rec[f"morn_{name}"] = _reached(morn, lvl, direction)
            rec[f"aft_{name}"] = _reached(aft, lvl, direction)
        rec["morn_reclaim_open"] = _reached(morn, o, up)
        rec["aft_reclaim_open"] = _reached(aft, o, up)
        if len(morn):
            rec["morn_ret_pct"] = 100 * (float(morn["adj_close"].iloc[-1]) - cut_px) / cut_px
            rec["morn_mae_pct"] = 100 * (float(morn["adj_low"].min()) - cut_px) / cut_px
            rec["morn_mfe_pct"] = 100 * (float(morn["adj_high"].max()) - cut_px) / cut_px
        rec["day_ret_pct"] = 100 * (float(g["adj_close"].iloc[-1]) - cut_px) / cut_px
        rec["day_mae_pct"] = 100 * (float(aft["adj_low"].min()) - cut_px) / cut_px
        rec["day_mfe_pct"] = 100 * (float(aft["adj_high"].max()) - cut_px) / cut_px
        rows.append(rec)
    t = pd.DataFrame(rows)
    return t.sort_values("session_date").reset_index(drop=True) if len(t) else t


# ---- individual parameter masks over the session table -------------------- #
def _mask_for(table: pd.DataFrame, fieldname: str, val, spec: PriceActionSpec) -> pd.Series:
    down = spec.direction == "down"
    T = lambda s: s.fillna(False).astype(bool)          # noqa: E731
    if fieldname == "ath_tolerance_pct":
        return table["distance_open_to_ath_pct"].abs() <= val
    if fieldname == "require_new_ath_at_open":
        return T(table["ath_at_open"])
    if fieldname == "require_open_above_pdh":
        return T(table["open_above_pdh"])
    if fieldname == "require_open_below_pdl":
        return T(table["open_below_pdl"])
    if fieldname == "gap_min_pct":
        return table["gap_pct"] >= val
    if fieldname == "gap_max_pct":
        return table["gap_pct"] <= val
    if fieldname == "on_location_min":
        return table["open_location_in_overnight_range"] >= val
    if fieldname == "on_location_max":
        return table["open_location_in_overnight_range"] <= val
    if fieldname == "vol_regimes":
        return table["daily_volatility_regime"].isin(val)
    if fieldname == "require_close_beyond_open":
        return T(table["pa_close_below_open" if down else "pa_close_above_open"])
    if fieldname == "close_location_max":
        loc = table["pa_close_location"]
        return loc <= val if down else loc >= (1.0 - val)
    if fieldname == "require_close_beyond_prior_bar":
        return T(table["pa_close_below_prior_low" if down else "pa_close_above_prior_high"])
    if fieldname == "require_monotonic_highs":
        return T(table["pa_monotonic_lower_highs" if down else "pa_monotonic_higher_lows"])
    if fieldname == "min_directional_bars":
        return table["pa_n_bear_bars" if down else "pa_n_bull_bars"] >= val
    if fieldname == "require_beyond_open_excursion":
        return T(table["pa_spike_up_reject" if down else "pa_spike_down_reject"])
    raise KeyError(fieldname)


# order in which conditions are evaluated / reported
PARAM_ORDER = [
    ("ath_tolerance_pct", "|open − ATH| ≤ {v}%"),
    ("require_new_ath_at_open", "new ATH at open"),
    ("require_open_above_pdh", "open > PDH"),
    ("require_open_below_pdl", "open < PDL"),
    ("gap_min_pct", "gap ≥ {v}%"),
    ("gap_max_pct", "gap ≤ {v}%"),
    ("on_location_min", "open ≥ {v} of ON range"),
    ("on_location_max", "open ≤ {v} of ON range"),
    ("vol_regimes", "regime in {v}"),
    ("require_close_beyond_open", "close beyond the open"),
    ("close_location_max", "close in the far {v} of the range"),
    ("require_close_beyond_prior_bar", "close beyond prior bar's extreme"),
    ("require_monotonic_highs", "monotonic highs/lows"),
    ("min_directional_bars", "≥{v} directional bars"),
    ("require_beyond_open_excursion", "poke beyond open then reject"),
]


def _active_params(spec: PriceActionSpec):
    """(fieldname, value, label) for every condition the spec actually enables."""
    out = []
    for fieldname, label in PARAM_ORDER:
        val = getattr(spec, fieldname)
        if val is None or val is False or (fieldname == "vol_regimes" and not val):
            continue
        out.append((fieldname, val, label.format(v=val)))
    return out


def apply_spec(table: pd.DataFrame, spec: PriceActionSpec = ATH_TRAP_SPEC) -> pd.Series:
    mask = pd.Series(True, index=table.index)
    for fieldname, val, _ in _active_params(spec):
        mask &= _mask_for(table, fieldname, val, spec).fillna(False)
    return mask


def build_funnel(table: pd.DataFrame, spec: PriceActionSpec = ATH_TRAP_SPEC) -> pd.DataFrame:
    """Sessions surviving each parameter, applied cumulatively in order."""
    rows = [{"step": "all sessions", "surviving": len(table), "cut": 0}]
    mask = pd.Series(True, index=table.index)
    prev = len(table)
    for fieldname, val, label in _active_params(spec):
        mask &= _mask_for(table, fieldname, val, spec).fillna(False)
        n = int(mask.sum())
        rows.append({"step": label, "surviving": n, "cut": prev - n})
        prev = n
    return pd.DataFrame(rows)


def scan(master: pd.DataFrame, daily: pd.DataFrame, spec: PriceActionSpec = ATH_TRAP_SPEC,
         cfg: Config = DEFAULT_CONFIG, table: pd.DataFrame | None = None
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matches, funnel).

    Pass a prebuilt ``table`` (from :func:`build_session_table`) to re-scan many
    specs without recomputing the per-bar geometry.
    """
    if table is None:
        table = build_session_table(master, daily, spec, cfg)
    if not len(table):
        return table, pd.DataFrame(columns=["step", "surviving", "cut"])
    matches = table[apply_spec(table, spec)].reset_index(drop=True)
    return matches, build_funnel(table, spec)


# --------------------------------------------------------------------------- #
# conditional probability report
# --------------------------------------------------------------------------- #
def conditional_report(matches: pd.DataFrame, spec: PriceActionSpec = ATH_TRAP_SPEC) -> pd.DataFrame:
    """P(level reached later | NOT reached by the decision bar).

    Levels already touched before the decision bar are removed from their own
    denominator -- you could not have entered ahead of them.
    """
    rows = []
    for name in LEVELS:
        pre, morn, aft = f"pre_{name}", f"morn_{name}", f"aft_{name}"
        if pre not in matches:
            continue
        untested = matches[~matches[pre].fillna(False)]
        n = len(untested)
        rows.append({
            "level": name,
            "tested_before_decision": int(matches[pre].sum()),
            "untested_at_decision": n,
            "reached_in_morning": int(untested[morn].sum()) if n else 0,
            "p_morning_%": round(100 * untested[morn].mean(), 1) if n else np.nan,
            "reached_by_close": int(untested[aft].sum()) if n else 0,
            "p_by_close_%": round(100 * untested[aft].mean(), 1) if n else np.nan,
        })
    if "morn_reclaim_open" in matches and len(matches):
        rows.append({
            "level": "OPEN (retrace)", "tested_before_decision": 0,
            "untested_at_decision": len(matches),
            "reached_in_morning": int(matches["morn_reclaim_open"].sum()),
            "p_morning_%": round(100 * matches["morn_reclaim_open"].mean(), 1),
            "reached_by_close": int(matches["aft_reclaim_open"].sum()),
            "p_by_close_%": round(100 * matches["aft_reclaim_open"].mean(), 1),
        })
    return pd.DataFrame(rows)


def performance_summary(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, lbl in (("morn_ret_pct", "morning return (decision→12:00)"),
                     ("morn_mae_pct", "morning MAE"), ("morn_mfe_pct", "morning MFE"),
                     ("day_ret_pct", "full-day return from decision"),
                     ("day_mae_pct", "day MAE"), ("day_mfe_pct", "day MFE")):
        if col not in matches:
            continue
        v = matches[col].dropna()
        if not len(v):
            continue
        rows.append({"metric": lbl, "n": len(v), "median": round(v.median(), 3),
                     "mean": round(v.mean(), 3), "negative_%": round(100 * (v < 0).mean(), 1),
                     "p25": round(v.quantile(.25), 3), "p75": round(v.quantile(.75), 3),
                     "worst": round(v.min(), 3), "best": round(v.max(), 3)})
    return pd.DataFrame(rows)
