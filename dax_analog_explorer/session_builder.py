"""Build sessions, the continuous 5-minute master series, and roll offsets.

Responsibilities
----------------
* Attach Berlin session date + minutes-since-cash-open to any bar frame.
* Resample FDAX 1m -> 5m (per contract, so no bin straddles a roll).
* Stitch DAX 5m (2000..2024-09) + FDAX-5m (2024-11..) into one continuous
  ``master_5m`` raw price series, carrying a ``roll_offset`` column so callers
  can form the *roll-adjusted* series (``adjusted = raw + roll_offset``) that the
  ATH engine needs -- without ever mixing raw levels across contracts.
* Compute the difference / "Panama" roll offsets (newest contract anchored raw,
  older contracts + FDAX gaps folded in; DAX segment carried on the same scale).
* Slice cash (09:00-17:30) and overnight (prev 17:30 -> 09:00) windows, and flag
  shortened sessions / holidays.

Roll offsets are estimated from the seam level jump (last cash bars of the
expiring contract vs first cash bars of the new contract).  Because the raw
contract files do not overlap, that jump also contains one weekend of real
market move -- an accepted, documented approximation (see README).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG


# --------------------------------------------------------------------------- #
# sessionizing helpers
# --------------------------------------------------------------------------- #
def attach_sessions(df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Add session_date, time-of-day minutes, minutes-since-open, cash flag."""
    df = df.copy()
    berlin = df["dt"].dt.tz_convert(ZoneInfo(cfg.tz))
    df["session_date"] = berlin.dt.normalize().dt.tz_localize(None)
    df["tod_min"] = berlin.dt.hour * 60 + berlin.dt.minute
    df["mso"] = df["tod_min"] - cfg.cash_open_minutes()   # minutes since cash open
    close_min = cfg.cash_close.hour * 60 + cfg.cash_close.minute
    df["is_cash"] = (df["tod_min"] >= cfg.cash_open_minutes()) & (df["tod_min"] <= close_min)
    return df


def cash_bars(df_sessioned: pd.DataFrame, date: pd.Timestamp,
              cfg: Config = DEFAULT_CONFIG, cutoff_min: int | None = None) -> pd.DataFrame:
    """Return cash-session bars for one date, optionally truncated at a cutoff.

    ``cutoff_min`` is minutes-since-open; bars with mso strictly greater are
    dropped.  This is the primary guard against future-data leakage.
    """
    date = pd.Timestamp(date).normalize()
    m = (df_sessioned["session_date"] == date) & (df_sessioned["is_cash"])
    out = df_sessioned.loc[m].sort_values("dt")
    if cutoff_min is not None:
        out = out[out["mso"] <= cutoff_min]
    return out


def overnight_bars(df_sessioned: pd.DataFrame, date: pd.Timestamp,
                   prev_date: pd.Timestamp, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Bars from the previous cash close (prev_date 17:30) up to date's 09:00."""
    tz = ZoneInfo(cfg.tz)
    start = pd.Timestamp(pd.Timestamp(prev_date).date(), tz=tz) + pd.Timedelta(
        hours=cfg.cash_close.hour, minutes=cfg.cash_close.minute)
    end = pd.Timestamp(pd.Timestamp(date).date(), tz=tz) + pd.Timedelta(
        hours=cfg.cash_open.hour, minutes=cfg.cash_open.minute)
    m = (df_sessioned["dt"] > start) & (df_sessioned["dt"] < end)
    return df_sessioned.loc[m].sort_values("dt")


# --------------------------------------------------------------------------- #
# FDAX 1m -> 5m
# --------------------------------------------------------------------------- #
def resample_fdax_to_5m(fdax: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Resample raw FDAX 1m to 5m, per contract, preserving contract metadata."""
    frames = []
    for cid, g in fdax.groupby("con_id"):
        gi = g.set_index("dt").sort_index()
        agg = pd.DataFrame({
            "open": gi["open"].resample("5min", label="left", closed="left").first(),
            "high": gi["high"].resample("5min", label="left", closed="left").max(),
            "low": gi["low"].resample("5min", label="left", closed="left").min(),
            "close": gi["close"].resample("5min", label="left", closed="left").last(),
            "volume": gi["volume"].resample("5min", label="left", closed="left").sum(),
        }).dropna(subset=["open"]).reset_index()
        agg["con_id"] = cid
        agg["contract_month"] = int(g["contract_month"].iloc[0])
        agg["source_file"] = g["source_file"].iloc[0]
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True).sort_values("dt").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# roll adjustment -> continuous BACK-ADJUSTED futures series
# --------------------------------------------------------------------------- #
# The raw FDAX contracts do not overlap, so the pure roll basis is not directly
# observable.  We support four methods (newest contract always anchored raw):
#
#   "carry"      (default) remove only the THEORETICAL cost-of-carry basis of the
#                new contract at the roll (immune to real weekend moves).  Most
#                correct here because it does not absorb genuine price change.
#   "ratio"      seam ratio  new_start/old_end  -> multiplicative, preserves % ;
#                data-driven but ABSORBS the real move across the roll weekend.
#   "difference" seam difference (Panama, additive) ; same real-move caveat.
#   "none"       raw stitch (no adjustment).
#
# ratio/carry are multiplicative (adjusted = raw * factor); difference is
# additive (adjusted = raw + offset).  ``factor``/``offset`` are populated for
# whichever family the method belongs to; the other is the identity.
@dataclass
class RollInfo:
    method: str
    factors: dict[int, float]                 # con_id -> multiplicative factor
    offsets: dict[int, float]                 # con_id -> additive offset
    dax_factor: float                         # for the DAX-5m segment
    dax_offset: float
    seams: list[dict]                         # per-roll report (both estimates)
    verification: dict                        # correctness checks
    multiplicative: bool                      # True for ratio/carry

    def adjust(self, raw: pd.Series, con_id_series: pd.Series,
               is_dax: pd.Series) -> pd.Series:
        """Apply the adjustment to a raw price column."""
        if self.multiplicative:
            fac = con_id_series.map(self.factors).astype("float64")
            fac = fac.where(~is_dax, self.dax_factor).fillna(1.0)
            return raw * fac
        off = con_id_series.map(self.offsets).astype("float64")
        off = off.where(~is_dax, self.dax_offset).fillna(0.0)
        return raw + off


def _contract_meta(fdax: pd.DataFrame, cfg: Config):
    """Ordered contracts with seam levels + expiry/start (for carry)."""
    fs = attach_sessions(fdax, cfg)
    fs = fs[fs["is_cash"]]
    order = (fdax.groupby("con_id")["contract_month"].first()
             .sort_values().index.tolist())

    def seam_level(cid, side):
        g = fs[fs["con_id"] == cid]
        day = g["session_date"].max() if side == "end" else g["session_date"].min()
        gg = g[g["session_date"] == day].sort_values("mso")
        if side == "end":
            gg = gg[gg["mso"] >= gg["mso"].max() - cfg.roll_seam_window_min]
        else:
            gg = gg[gg["mso"] <= gg["mso"].min() + cfg.roll_seam_window_min]
        return float(gg["close"].median())

    meta = {}
    for cid in order:
        g = fdax[fdax["con_id"] == cid]
        gs = fs[fs["con_id"] == cid]
        exp = pd.to_datetime(str(int(g["expiry"].iloc[0])), format="%Y%m%d")
        start = pd.Timestamp(gs["session_date"].min())
        meta[cid] = {
            "contract_month": int(g["contract_month"].iloc[0]),
            "end_level": seam_level(cid, "end"),
            "start_level": seam_level(cid, "start"),
            "expiry": exp, "start_date": start,
            "dte_at_start": max((exp - start).days, 1),
        }
    return order, meta


def compute_roll_adjustment(fdax: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> RollInfo:
    order, meta = _contract_meta(fdax, cfg)
    method = "none" if not cfg.roll_adjust_enabled else cfg.roll_adjust_method

    # per-seam estimates (both the measured seam gap and the carry model)
    seams, seam_ratio, seam_diff, carry_ratio = [], {}, {}, {}
    for a, b in zip(order[:-1], order[1:]):
        end_a, start_b = meta[a]["end_level"], meta[b]["start_level"]
        gap = start_b - end_a
        carry = end_a * cfg.carry_annual_rate * meta[b]["dte_at_start"] / 365.0
        seam_ratio[a] = start_b / end_a
        seam_diff[a] = gap
        carry_ratio[a] = 1.0 + (carry / end_a)
        seams.append({
            "old": meta[a]["contract_month"], "new": meta[b]["contract_month"],
            "old_end_level": round(end_a, 2), "new_start_level": round(start_b, 2),
            "seam_gap_pts": round(gap, 2), "seam_gap_pct": round(100 * gap / end_a, 3),
            "carry_pts": round(carry, 2), "carry_pct": round(100 * carry / end_a, 3),
            "real_move_pts_est": round(gap - carry, 2),
        })

    factors = {cid: 1.0 for cid in order}
    offsets = {cid: 0.0 for cid in order}
    multiplicative = method in ("ratio", "carry")

    if method != "none":
        per = carry_ratio if method == "carry" else (seam_ratio if method == "ratio" else None)
        # cumulate from newest (identity) back to oldest
        for k in range(len(order) - 2, -1, -1):
            a, b = order[k], order[k + 1]
            if multiplicative:
                factors[a] = factors[b] * per[a]
            else:
                offsets[a] = offsets[b] + seam_diff[a]

    dax_factor = factors[order[0]]
    dax_offset = offsets[order[0]]

    ri = RollInfo(method, factors, offsets, dax_factor, dax_offset,
                  seams, {}, multiplicative)
    ri.verification = _verify_adjustment(fdax, meta, order, ri, cfg)
    return ri


def _verify_adjustment(fdax, meta, order, ri: RollInfo, cfg) -> dict:
    """Checks that back-adjustment is well-formed (surfaced to the user)."""
    # residual seam discontinuity AFTER adjustment
    residuals = []
    for a, b in zip(order[:-1], order[1:]):
        if ri.multiplicative:
            adj_end_a = meta[a]["end_level"] * ri.factors[a]
            adj_start_b = meta[b]["start_level"] * ri.factors[b]
        else:
            adj_end_a = meta[a]["end_level"] + ri.offsets[a]
            adj_start_b = meta[b]["start_level"] + ri.offsets[b]
        residuals.append(100 * (adj_start_b - adj_end_a) / adj_end_a)
    residuals = np.array(residuals)
    return {
        "method": ri.method,
        "newest_contract_anchored_raw": bool(
            (ri.multiplicative and abs(ri.factors[order[-1]] - 1.0) < 1e-12)
            or (not ri.multiplicative and abs(ri.offsets[order[-1]]) < 1e-9)),
        "seam_residual_pct_max": round(float(np.max(np.abs(residuals))), 4) if len(residuals) else 0.0,
        "seam_residual_pct_mean": round(float(np.mean(residuals)), 4) if len(residuals) else 0.0,
        "n_seams": len(residuals),
        # for carry: residual == the real weekend move left in place (as intended)
        "residual_is_real_move": ri.method == "carry",
        "preserves_pct_returns": ri.multiplicative,
        "note": {
            "carry": "removes theoretical carry only; residual per seam is the real "
                     "weekend move (correctly left in place).",
            "ratio": "seam-matched multiplicative; residual ~0 but absorbs the real "
                     "weekend move into the factor.",
            "difference": "seam-matched additive (Panama); absorbs the real weekend move.",
            "none": "raw stitch; contains full carry roll steps.",
        }[ri.method],
    }


# backward-compatible alias
def compute_roll_offsets(fdax: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> RollInfo:
    return compute_roll_adjustment(fdax, cfg)


# --------------------------------------------------------------------------- #
# master continuous 5m series
# --------------------------------------------------------------------------- #
def build_master_5m(dax5: pd.DataFrame, fdax1: pd.DataFrame,
                    cfg: Config = DEFAULT_CONFIG) -> tuple[pd.DataFrame, RollInfo]:
    """Stitch DAX-5m + FDAX-5m into one continuous series.

    Emits BOTH raw OHLC (genuine traded levels, for charts) and back-adjusted
    ``adj_*`` OHLC (continuous, newest contract anchored raw) which the ATH engine
    and any cross-day price comparison must use so no raw levels are ever mixed
    across contracts.
    """
    roll = compute_roll_adjustment(fdax1, cfg)
    fdax5 = resample_fdax_to_5m(fdax1, cfg)

    dax = dax5.copy()
    dax["con_id"] = pd.NA
    dax["contract_month"] = pd.NA
    dax["instrument"] = "DAX5m"

    f = fdax5.copy()
    f["instrument"] = "FDAX5m"
    if "source_file" not in f:
        f["source_file"] = pd.NA

    cols = ["dt", "open", "high", "low", "close", "volume",
            "con_id", "contract_month", "instrument", "source_file"]
    master = pd.concat([dax[cols], f[cols]], ignore_index=True)
    master = master.sort_values("dt", kind="mergesort").reset_index(drop=True)

    is_dax = master["instrument"] == "DAX5m"
    for c in ("open", "high", "low", "close"):
        master[f"adj_{c}"] = roll.adjust(master[c], master["con_id"], is_dax)
    # store the per-bar scalar too (handy for the audit / debugging)
    master["adj_factor"] = (master["adj_close"] / master["close"]).where(master["close"] != 0, 1.0)
    return master, roll


# --------------------------------------------------------------------------- #
# trading calendar / holidays / shortened sessions
# --------------------------------------------------------------------------- #
def build_calendar(master_sessioned: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """One row per session date: bar counts, cash bar counts, shortened flag."""
    df = master_sessioned
    cash = df[df["is_cash"]]
    g = df.groupby("session_date")
    cal = pd.DataFrame({
        "n_bars": g.size(),
        "n_cash_bars": cash.groupby("session_date").size(),
        "instrument": g["instrument"].agg(lambda s: s.iloc[0]),
    }).reset_index()
    cal["n_cash_bars"] = cal["n_cash_bars"].fillna(0).astype(int)
    cal["weekday"] = cal["session_date"].dt.day_name()
    # expected full cash bars for a 09:00-17:30 5m session
    full = ((cfg.cash_close.hour * 60 + cfg.cash_close.minute) - cfg.cash_open_minutes()) // 5 + 1
    med = cal["n_cash_bars"].median()
    cal["shortened_session"] = cal["n_cash_bars"] < 0.6 * max(med, full * 0.5)
    cal["is_weekend"] = cal["session_date"].dt.weekday >= 5
    return cal.sort_values("session_date").reset_index(drop=True)


def previous_trading_date(cal: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    dates = cal["session_date"].tolist()
    date = pd.Timestamp(date).normalize()
    prior = [d for d in dates if d < date]
    return prior[-1] if prior else None


# --------------------------------------------------------------------------- #
# per-session OHLC (shared backbone for ATH + daily features)
# --------------------------------------------------------------------------- #
def build_daily_ohlc(master: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """One row per session date with cash / overnight / full-day OHLC on both raw
    and back-adjusted prices.  Overnight = prev cash close .. this cash open,
    assigned with a forward as-of join on the 09:00 session boundary.
    """
    ms = attach_sessions(master, cfg).sort_values("dt").reset_index(drop=True)
    tz = ZoneInfo(cfg.tz)
    open_min = cfg.cash_open_minutes()

    # ---- cash-session OHLC (raw + adj) ----
    cash = ms[ms["is_cash"]]
    def ohlc(frame, prefix):
        gg = frame.sort_values("dt").groupby("session_date")
        d = pd.DataFrame({
            f"{prefix}_open": gg["open"].first(),
            f"{prefix}_high": gg["high"].max(),
            f"{prefix}_low": gg["low"].min(),
            f"{prefix}_close": gg["close"].last(),
            f"{prefix}_open_adj": gg["adj_open"].first(),
            f"{prefix}_high_adj": gg["adj_high"].max(),
            f"{prefix}_low_adj": gg["adj_low"].min(),
            f"{prefix}_close_adj": gg["adj_close"].last(),
        })
        return d
    cash_ohlc = ohlc(cash, "cash")
    cash_ohlc["n_cash_bars"] = cash.groupby("session_date").size()

    # ---- full calendar-day high/low (adjusted) for prior-ATH ----
    day = ms.groupby("session_date").agg(
        day_high_adj=("adj_high", "max"), day_low_adj=("adj_low", "min"),
        instrument=("instrument", "first"))

    # ---- overnight OHLC via forward as-of to the 09:00 boundary ----
    sessions = pd.DatetimeIndex(sorted(ms["session_date"].unique()))
    open_ts = pd.DataFrame({
        "session_date": sessions,
        "open_ts": [pd.Timestamp(d.date(), tz=tz) + pd.Timedelta(minutes=open_min)
                    for d in sessions],
    }).sort_values("open_ts")
    non_cash = ms[~ms["is_cash"]].copy().sort_values("dt")
    non_cash = pd.merge_asof(non_cash, open_ts[["open_ts", "session_date"]].rename(
        columns={"session_date": "overnight_of"}),
        left_on="dt", right_on="open_ts", direction="forward")
    on = non_cash.dropna(subset=["overnight_of"]).groupby("overnight_of")
    overnight = pd.DataFrame({
        "overnight_open": on["open"].first(), "overnight_high": on["high"].max(),
        "overnight_low": on["low"].min(), "overnight_close": on["close"].last(),
        "overnight_open_adj": on["adj_open"].first(), "overnight_high_adj": on["adj_high"].max(),
        "overnight_low_adj": on["adj_low"].min(), "overnight_close_adj": on["adj_close"].last(),
        "n_overnight_bars": on.size(),
    })
    overnight.index.name = "session_date"

    out = cash_ohlc.join(day, how="left").join(overnight, how="left").reset_index()
    out = out.sort_values("session_date").reset_index(drop=True)
    out["prev_session_date"] = out["session_date"].shift(1)
    out["weekday"] = out["session_date"].dt.day_name()
    return out
