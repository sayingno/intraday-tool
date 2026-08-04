"""Load and standardize the raw intraday files into a common schema.

Canonical schema (both instruments) -- one bar per row:

    dt           tz-aware pandas Timestamp in Europe/Berlin (bar OPEN time)
    open high low close   float64
    volume       float64
    source_file  str      original filename the bar came from

FDAX rows additionally carry: contract_month, expiry, con_id, local_symbol,
average, bar_count.

Timezone handling is the whole point of this module:

* DAX 5m timestamps are America/Chicago wall-clock -> localized in Chicago
  (DST-aware, so the ~5-6 weeks/year where the US/EU DST calendars disagree get
  the correct 6h vs 7h offset) and converted to Europe/Berlin.
* FDAX 1m timestamps are already Europe/Berlin wall-clock -> localized directly.

Nothing here assumes a fixed hour offset.
"""
from __future__ import annotations

import glob
import io
import zipfile
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG

CANONICAL_COLS = ["dt", "open", "high", "low", "close", "volume", "source_file"]


@dataclass
class LoadResult:
    """A cleaned frame plus the bookkeeping the audit wants to report."""
    df: pd.DataFrame
    n_raw: int
    n_exact_duplicates: int
    n_conflicting_duplicates: int
    conflicts: pd.DataFrame
    localize_fallback_used: bool
    source_files: list[str]


# --------------------------------------------------------------------------- #
# timezone helpers
# --------------------------------------------------------------------------- #
def localize_to_berlin(naive: pd.Series, source_tz: str) -> tuple[pd.DatetimeIndex, bool]:
    """Localize naive wall-clock timestamps in ``source_tz`` and convert to Berlin.

    Returns (tz_aware_index_in_berlin, fallback_used).  ``ambiguous='infer'`` is
    tried first (correct at DST fall-back when both folds are present); if the
    series can't be inferred (e.g. missing bars around a transition) we fall back
    to treating ambiguous local times as DST, which at worst mislabels the single
    repeated overnight hour once a year -- immaterial to cash-session analysis.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(naive))
    src = ZoneInfo(source_tz)
    berlin = ZoneInfo("Europe/Berlin")
    fallback = False
    try:
        loc = idx.tz_localize(src, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        fallback = True
        loc = idx.tz_localize(src, ambiguous=True, nonexistent="shift_forward")
    return loc.tz_convert(berlin), fallback


# --------------------------------------------------------------------------- #
# duplicate handling
# --------------------------------------------------------------------------- #
def _split_duplicates(df: pd.DataFrame, key: list[str],
                      ohlcv=("open", "high", "low", "close", "volume")):
    """Drop exact-duplicate bars, flag conflicting ones (same key, diff OHLCV).

    Returns (clean_df, n_exact, n_conflicting, conflicts_df).
    """
    ohlcv = list(ohlcv)
    # exact duplicates: identical across key + all OHLCV
    exact_mask = df.duplicated(subset=key + ohlcv, keep="first")
    n_exact = int(exact_mask.sum())
    df = df.loc[~exact_mask].copy()

    # conflicts: same key remains duplicated but OHLCV differ
    dup_key = df.duplicated(subset=key, keep=False)
    conflicts = df.loc[dup_key].sort_values(key)
    n_conflicting = int(conflicts[key].drop_duplicates().shape[0])
    # keep the first occurrence for a deterministic series
    df = df.loc[~df.duplicated(subset=key, keep="first")].copy()
    return df, n_exact, n_conflicting, conflicts


# --------------------------------------------------------------------------- #
# DAX 5-minute  (Chicago -> Berlin)
# --------------------------------------------------------------------------- #
def load_dax_5m(cfg: Config = DEFAULT_CONFIG) -> LoadResult:
    zip_path = cfg.paths.dax_zip
    with zipfile.ZipFile(zip_path) as zf:
        name = cfg.paths.dax_csv_name
        if name not in zf.namelist():
            name = zf.namelist()[0]
        raw = zf.read(name)
    df = pd.read_csv(
        io.BytesIO(raw), sep=";", header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={"date": str, "time": str},
    )
    n_raw = len(df)
    naive = pd.to_datetime(df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M:%S")
    berlin, fallback = localize_to_berlin(naive, cfg.source_tz_dax)
    out = pd.DataFrame({
        "dt": berlin,
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["volume"].astype("float64"),
        "source_file": name,
    }).sort_values("dt", kind="mergesort").reset_index(drop=True)

    out, n_exact, n_conf, conflicts = _split_duplicates(out, key=["dt"])
    out = out.sort_values("dt", kind="mergesort").reset_index(drop=True)
    return LoadResult(out, n_raw, n_exact, n_conf, conflicts, fallback, [name])


# --------------------------------------------------------------------------- #
# FDAX 1-minute  (already Berlin) -- raw quarterly contracts
# --------------------------------------------------------------------------- #
FDAX_COLMAP = {
    "Datetime": "datetime", "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Volume": "volume", "ContractMonth": "contract_month",
    "Expiry": "expiry", "ConId": "con_id", "LocalSymbol": "local_symbol",
    "Average": "average", "BarCount": "bar_count",
}


def load_fdax_1m(cfg: Config = DEFAULT_CONFIG) -> LoadResult:
    files = sorted(glob.glob(str(cfg.paths.repo_root / cfg.paths.fdax_glob)))
    if not files:
        raise FileNotFoundError(f"No FDAX files matched {cfg.paths.fdax_glob}")
    frames = []
    n_raw = 0
    for f in files:
        d = pd.read_csv(f)
        n_raw += len(d)
        d = d.rename(columns=FDAX_COLMAP)
        d["source_file"] = f.split("/")[-1]
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)

    berlin, fallback = localize_to_berlin(raw["datetime"], cfg.source_tz_fdax)
    out = pd.DataFrame({
        "dt": berlin,
        "open": raw["open"].astype("float64"),
        "high": raw["high"].astype("float64"),
        "low": raw["low"].astype("float64"),
        "close": raw["close"].astype("float64"),
        "volume": raw["volume"].astype("float64"),
        "source_file": raw["source_file"],
        "contract_month": raw["contract_month"].astype("int64"),
        "expiry": raw["expiry"].astype("int64"),
        "con_id": raw["con_id"].astype("int64"),
        "local_symbol": raw["local_symbol"].astype(str),
        "average": raw["average"].astype("float64"),
        "bar_count": raw["bar_count"].astype("float64"),
    }).sort_values("dt", kind="mergesort").reset_index(drop=True)

    # dedup within the same contract (con_id); different contracts sharing a
    # timestamp are legitimately distinct instruments, not duplicates.
    out, n_exact, n_conf, conflicts = _split_duplicates(out, key=["dt", "con_id"])
    out = out.sort_values(["dt", "con_id"], kind="mergesort").reset_index(drop=True)
    return LoadResult(out, n_raw, n_exact, n_conf, conflicts, fallback,
                      [f.split("/")[-1] for f in files])


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def add_session_date(df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Attach the Berlin calendar date of each bar (used for grouping)."""
    df = df.copy()
    df["session_date"] = df["dt"].dt.tz_convert(ZoneInfo(cfg.tz)).dt.date
    return df
