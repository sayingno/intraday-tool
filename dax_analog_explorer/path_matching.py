"""Path-shape matching: align intraday paths at the cash open and compare shape.

Every path is rebased so the cash open is zero, expressed in the requested unit
(points / percent / prior-day-range / ATR), and truncated at the observation
cutoff -- no post-cutoff price ever enters a path used for matching.

Similarity primitives: Pearson correlation, Euclidean distance, cosine
similarity, and an optional pure-numpy Dynamic Time Warping distance (added on
top of the transparent feature+correlation baseline, as the spec sequences it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from . import session_builder as sb


def build_cash_groups(master: pd.DataFrame, cfg: Config = DEFAULT_CONFIG,
                      max_min: int | None = None) -> dict:
    ms = sb.attach_sessions(master, cfg)
    m = ms["is_cash"] & (ms["mso"] >= 0)
    if max_min is not None:
        m &= ms["mso"] <= max_min
    cash = ms[m]
    return dict(tuple(cash.groupby("session_date")))


def session_path(group: pd.DataFrame, cutoff_min: int, unit: str,
                 ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (mso_grid, values) for one session, rebased to open=0."""
    b = group[(group["mso"] >= 0) & (group["mso"] <= cutoff_min)].sort_values("mso")
    if len(b) == 0:
        return np.array([]), np.array([])
    open_px = ref["cash_open"]
    delta = b["adj_close"].to_numpy() - open_px
    if unit == "pct":
        val = 100.0 * delta / open_px
    elif unit == "prior_range":
        val = delta / ref.get("prev_day_range", np.nan)
    elif unit == "atr":
        val = delta / ref.get("atr", np.nan)
    else:  # points
        val = delta
    return b["mso"].to_numpy(), val


def align_on_grid(mso: np.ndarray, val: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Forward-fill a path onto a common minute grid (0..cutoff)."""
    if len(mso) == 0:
        return np.full(len(grid), np.nan)
    s = pd.Series(val, index=mso).reindex(
        sorted(set(mso).union(grid))).sort_index().ffill()
    return s.reindex(grid).to_numpy()


def build_path_matrix(groups: dict, dates: list, cutoff_min: int, unit: str,
                      refs: dict, step: int = 5) -> tuple[np.ndarray, np.ndarray, list]:
    grid = np.arange(0, cutoff_min + 1, step)
    rows, kept = [], []
    for d in dates:
        if d not in groups or d not in refs:
            continue
        mso, val = session_path(groups[d], cutoff_min, unit, refs[d])
        if len(mso) == 0:
            continue
        rows.append(align_on_grid(mso, val, grid))
        kept.append(d)
    mat = np.vstack(rows) if rows else np.empty((0, len(grid)))
    return grid, mat, kept


# --------------------------------------------------------------------------- #
# similarity primitives
# --------------------------------------------------------------------------- #
def _prep(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def path_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _prep(a, b)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def path_euclidean(a: np.ndarray, b: np.ndarray, normalize: bool = True) -> float:
    a, b = _prep(a, b)
    if len(a) == 0:
        return np.nan
    d = np.sqrt(np.mean((a - b) ** 2))       # RMS so length-independent
    return float(d)


def path_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _prep(a, b)
    if len(a) == 0 or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return np.nan
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    """Pure-numpy DTW (Sakoe-Chiba band optional).  Used only when requested."""
    a, b = _prep(a, b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    w = max(window or max(n, m), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - w), min(m, i + w)
        for j in range(jlo, jhi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def path_distance(a: np.ndarray, b: np.ndarray, method: str = "correlation") -> float:
    """Map a chosen primitive to a DISTANCE (0 = identical, larger = different)."""
    if method == "correlation":
        c = path_correlation(a, b)
        return np.nan if pd.isna(c) else (1.0 - c) / 2.0
    if method == "cosine":
        c = path_cosine(a, b)
        return np.nan if pd.isna(c) else (1.0 - c) / 2.0
    if method == "dtw":
        return dtw_distance(a, b)
    return path_euclidean(a, b)   # already a distance
