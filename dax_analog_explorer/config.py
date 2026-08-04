"""Central configuration for the DAX Analog Day Explorer.

Everything tunable lives here as a single ``Config`` dataclass so the Streamlit
UI, the CLI preprocessor and the tests all share one source of truth.  Nothing
in this file assumes a timezone silently -- the source timezones are declared
explicitly and verified by :mod:`dax_analog_explorer.data_audit`.

Key facts established by the data audit (see README):

* ``dax-5m.csv``  -> DAX **futures** 5-minute, timestamps in **America/Chicago**,
  covering 2000-05-29 .. 2024-09-18.  (Verified from the intraday volume
  profile: 2019-2024 open-auction volume peaks at 02:00 Chicago = 09:00 Berlin.)
* ``FDAX_1min_*.csv`` -> FDAX **futures** 1-minute, timestamps already in
  **Europe/Berlin** (Eurex), raw quarterly contracts 2024-11-01 .. 2026-05-19.

Per the user's instruction the whole project treats price as *futures* and the
all-time-high is computed on a single roll-adjusted continuous futures series.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repo root = parent of this package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass
class Paths:
    repo_root: Path = REPO_ROOT
    dax_zip: Path = REPO_ROOT / "dax-5m.zip"
    # inside the zip:
    dax_csv_name: str = "dax-5m.csv"
    # raw FDAX contract files live at the repo root:
    fdax_glob: str = "FDAX_1min_*.csv"
    processed_dir: Path = PROCESSED_DIR

    # output parquet artifacts (the 4 spec-named + the merged master + audit)
    cleaned_dax_5m: Path = PROCESSED_DIR / "cleaned_dax_5m.parquet"
    cleaned_fdax_1m: Path = PROCESSED_DIR / "cleaned_fdax_1m.parquet"
    master_5m: Path = PROCESSED_DIR / "master_5m.parquet"
    daily_features: Path = PROCESSED_DIR / "daily_features.parquet"
    opening_path_features: Path = PROCESSED_DIR / "opening_path_features.parquet"
    audit_report_json: Path = PROCESSED_DIR / "audit_report.json"
    audit_report_txt: Path = PROCESSED_DIR / "audit_report.txt"


# --------------------------------------------------------------------------- #
# Main configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # -- timezones (declared, then verified in data_audit) -----------------
    tz: str = "Europe/Berlin"
    source_tz_dax: str = "America/Chicago"
    source_tz_fdax: str = "Europe/Berlin"

    # -- session definitions (all configurable) ----------------------------
    cash_open: dt.time = dt.time(9, 0)
    cash_close: dt.time = dt.time(17, 30)
    # overnight = previous cash_close .. current cash_open (futures bars in between)
    us_cash_open: dt.time = dt.time(15, 30)  # US equity open in Berlin time
    midday: dt.time = dt.time(12, 0)

    # -- observation windows / cutoffs -------------------------------------
    opening_windows_min: tuple[int, ...] = (5, 15, 30, 60)
    default_cutoffs: tuple[str, ...] = ("09:15", "09:30", "10:00", "10:30")
    default_cutoff: str = "10:30"
    weakness_deadline: str = "10:30"   # "before ~10:30" in the flagship pattern

    # bar size of the merged/master series used for path matching & features
    master_bar_minutes: int = 5

    # -- ATH engine --------------------------------------------------------
    ath_tolerances_pct: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    default_ath_tolerance_pct: float = 0.25
    # roll adjustment for stitching raw FDAX quarterly contracts into a
    # continuous series before any all-time-high comparison.
    roll_adjust_method: str = "carry"        # {"carry", "ratio", "difference", "none"}
    roll_adjust_enabled: bool = True
    # window (minutes into the cash session) used to measure the seam level
    roll_seam_window_min: int = 30
    # annualized cost-of-carry used by the "carry" roll method (DAX is a total-
    # return index, so fair-value basis ~= financing rate; no dividend drag).
    carry_annual_rate: float = 0.025
    # FDAX raw contracts do not overlap, so a continuous futures ATH is only
    # trustworthy on the *adjusted* series; expose a flag for transparency.
    enable_raw_fut_ath: bool = False

    # -- rolling / regime windows (all use PRIOR sessions only) -------------
    atr_window: int = 14
    vol_window: int = 20
    regime_lookback: int = 60
    regime_low_q: float = 0.33
    regime_high_q: float = 0.66

    # -- pattern-feature parameters ----------------------------------------
    swing_lookback: int = 2                 # bars each side for swing hi/lo
    expansion_bar_multiple: float = 1.5     # bar range > mult * recent median => "expansion"
    expansion_median_window: int = 10       # bars for the recent-median bar range
    meaningful_move_atr_frac: float = 0.15  # "meaningful" lower-high / pullback threshold
    overlap_definition: str = "range"       # bar-to-bar overlap basis

    # thresholds used by the flagship "weak follow-through" descriptor
    # (defaults only -- every one is overridable from the UI)
    first_bar_range_pctl: float = 80.0      # "large" first bar >= this historical pctl
    first_bar_close_top_frac: float = 0.80  # close in top 20% of its range
    weak_extension_ratio_max: float = 0.50
    weak_overlap_ratio_min: float = 0.50
    weak_efficiency_max: float = 0.40

    # -- similarity weights (4 blocks; must sum ~1.0) ----------------------
    # each block weight and its intra-block feature weights are configurable.
    weight_context: float = 0.30
    weight_first_bar: float = 0.20
    weight_post_spike: float = 0.30
    weight_weakness: float = 0.20
    # path-correlation gets blended into the post-spike block:
    path_corr_weight_in_post: float = 0.40

    context_features: tuple[str, ...] = (
        "distance_open_to_ath_pct", "overnight_return_pct",
        "overnight_range_normalized", "gap_pct", "open_vs_pdh_pct",
    )
    first_bar_features: tuple[str, ...] = (
        "first_bar_return", "first_bar_range_normalized",
        "first_bar_body_to_range", "first_bar_close_location",
        "first_bar_breaks_overnight_high",
    )
    post_spike_features: tuple[str, ...] = (
        "extension_ratio", "post_spike_pullback_ratio",
        "post_spike_overlap_ratio", "post_spike_efficiency",
        "number_of_new_highs_after_first_bar",
    )
    weakness_features: tuple[str, ...] = (
        "first_meaningful_lower_high_min", "first_bearish_expansion_min",
        "first_break_first15m_low_min", "time_below_open_frac",
    )

    # -- outcome horizons (minutes after the observation cutoff) -----------
    outcome_horizons_min: tuple[int, ...] = (15, 30, 60)

    # -- progressive matching relaxation ladder ----------------------------
    # applied in order until we reach at least `min_sample` matches.
    min_sample: int = 20
    relax_ladder: tuple[dict[str, Any], ...] = field(default_factory=lambda: (
        {"name": "ath_tolerance", "to": 0.50, "desc": "ATH tolerance widened to 0.50%"},
        {"name": "weakness_deadline", "to": "10:45", "desc": "weakness timing widened by 15 min"},
        {"name": "ath_tolerance", "to": 1.00, "desc": "ATH tolerance widened to 1.00%"},
        {"name": "drop_gap", "to": None, "desc": "gap-direction condition removed"},
        {"name": "drop_regime", "to": None, "desc": "volatility-regime condition removed"},
    ))

    # -- misc --------------------------------------------------------------
    default_n_results: int = 20
    paths: Paths = field(default_factory=Paths)

    # ------------------------------------------------------------------ #
    def cutoff_time(self, s: str | None = None) -> dt.time:
        """Parse an "HH:MM" cutoff string into a ``datetime.time``."""
        s = s or self.default_cutoff
        h, m = s.split(":")
        return dt.time(int(h), int(m))

    def cash_open_minutes(self) -> int:
        return self.cash_open.hour * 60 + self.cash_open.minute

    def minutes_since_open(self, t: dt.time) -> int:
        return (t.hour * 60 + t.minute) - self.cash_open_minutes()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("paths", None)
        # times -> "HH:MM"
        for k, v in list(d.items()):
            if isinstance(v, dt.time):
                d[k] = v.strftime("%H:%M")
        return d

    def copy_with(self, **overrides: Any) -> "Config":
        """Return a new Config with the given fields overridden (UI helper)."""
        base = {f: getattr(self, f) for f in self.__dataclass_fields__ if f != "paths"}
        for k, v in overrides.items():
            if isinstance(getattr(self, k, None), dt.time) and isinstance(v, str):
                hh, mm = v.split(":")
                v = dt.time(int(hh), int(mm))
            base[k] = v
        return Config(paths=self.paths, **base)


DEFAULT_CONFIG = Config()
