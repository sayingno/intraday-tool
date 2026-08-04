"""Shared synthetic fixtures — small, deterministic, no disk I/O."""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

BER = ZoneInfo("Europe/Berlin")


def cash_frame(date="2024-06-03", ohlc=None, start_hm=(9, 0), step=5):
    """Build a sessioned cash-bar frame (adj_* == raw) from a list of (o,h,l,c)."""
    if ohlc is None:
        ohlc = [(100, 101, 99, 100.5)]
    d = pd.Timestamp(date)
    rows = []
    for i, (o, h, l, c) in enumerate(ohlc):
        t = pd.Timestamp(d.date(), tz=BER) + pd.Timedelta(hours=start_hm[0],
                                                          minutes=start_hm[1] + i * step)
        rows.append({"dt": t, "open": o, "high": h, "low": l, "close": c,
                     "adj_open": o, "adj_high": h, "adj_low": l, "adj_close": c,
                     "volume": 100.0})
    df = pd.DataFrame(rows)
    df["session_date"] = pd.Timestamp(d.date())
    df["tod_min"] = start_hm[0] * 60 + start_hm[1] + np.arange(len(df)) * step
    df["mso"] = df["tod_min"] - 9 * 60
    df["is_cash"] = True
    return df


def default_ref(cash_open=100.0, **kw):
    ref = {"cash_open": cash_open, "pdh": 99.0, "pdl": 95.0, "prev_close": 98.0,
           "overnight_high": 100.2, "overnight_low": 97.0, "prior_ath": 101.0,
           "atr": 2.0, "prev_day_range": 3.0}
    ref.update(kw)
    return ref


def fdax_synthetic():
    """3 quarterly contracts with a known +10-point carry-ish seam each roll."""
    frames = []
    base = {"c1": (100.0, 20240315, 111, "MAR"),
            "c2": (110.0, 20240621, 222, "JUN"),
            "c3": (120.0, 20240920, 333, "SEP")}
    starts = {"c1": "2024-01-02", "c2": "2024-03-18", "c3": "2024-06-24"}
    ends = {"c1": "2024-03-15", "c2": "2024-06-21", "c3": "2024-09-20"}
    cm = {"c1": 202403, "c2": 202406, "c3": 202409}
    for key, (lvl, exp, cid, sym) in base.items():
        days = pd.bdate_range(starts[key], ends[key])
        for dd in days:
            for hm in [(9, 0), (13, 0), (17, 30)]:
                t = pd.Timestamp(dd.date(), tz=BER) + pd.Timedelta(hours=hm[0], minutes=hm[1])
                frames.append({"dt": t, "open": lvl, "high": lvl + 1, "low": lvl - 1,
                               "close": lvl, "volume": 50.0, "contract_month": cm[key],
                               "expiry": exp, "con_id": cid, "local_symbol": sym,
                               "average": lvl, "bar_count": 5.0,
                               "source_file": f"{key}.csv"})
    return pd.DataFrame(frames)


@pytest.fixture(scope="session")
def cfg():
    from dax_analog_explorer.config import Config
    return Config()
