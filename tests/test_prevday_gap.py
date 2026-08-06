"""Previous-day levels, gaps and open-vs-range flags (integration on a mini master)."""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dax_analog_explorer.config import Config
from dax_analog_explorer import feature_engineering as fe

BER = ZoneInfo("Europe/Berlin")


def _master(days):
    rows = []
    for date, (O, H, L, C) in days.items():
        d = pd.Timestamp(date)
        # 17:25 is the LAST bar of a 09:00-17:30 session (it spans 17:25-17:30);
        # a bar stamped 17:30 would already be past the close.
        for hm, bar in ((( 9, 0), (O, H, O, O)), ((17, 25), (O, O, L, C))):
            t = pd.Timestamp(d.date(), tz=BER) + pd.Timedelta(hours=hm[0], minutes=hm[1])
            o, h, l, c = bar
            rows.append({"dt": t, "open": o, "high": h, "low": l, "close": c,
                         "adj_open": o, "adj_high": h, "adj_low": l, "adj_close": c,
                         "volume": 100.0, "con_id": pd.NA, "contract_month": pd.NA,
                         "instrument": "DAX5m", "source_file": "x"})
    return pd.DataFrame(rows)


DAYS = {
    "2024-06-03": (100, 105, 99, 104),
    "2024-06-04": (104, 108, 103, 100),
    "2024-06-05": (105, 106, 101, 102),
    "2024-06-06": (101, 104, 100, 103),
}


def test_previous_day_levels_and_gap():
    daily = fe.build_daily_features(_master(DAYS), Config())
    d = daily.set_index("session_date")

    r3 = d.loc[pd.Timestamp("2024-06-05")]
    assert r3["previous_close"] == 100
    assert r3["previous_day_high"] == 108
    assert r3["previous_day_low"] == 103
    assert r3["gap_points"] == 5
    assert abs(r3["gap_pct"] - 5.0) < 1e-9
    assert bool(r3["open_above_pdh"]) is False
    assert bool(r3["open_below_pdl"]) is False
    assert bool(r3["open_inside_previous_range"]) is True

    r4 = d.loc[pd.Timestamp("2024-06-06")]
    assert r4["previous_day_high"] == 106
    assert r4["gap_points"] == -1


def test_open_above_pdh_flag():
    days = dict(DAYS); days["2024-06-06"] = (110, 112, 109, 111)  # opens above prev high 106
    daily = fe.build_daily_features(_master(days), Config()).set_index("session_date")
    assert bool(daily.loc[pd.Timestamp("2024-06-06"), "open_above_pdh"]) is True


def test_cash_close_uses_price_at_1730_not_after():
    """The bar stamped 17:30 spans 17:30-17:35 and is PAST the cash close.

    Regression: `is_cash` used `<= 17:30`, so previous_close picked up the
    post-close price and every gap_pct was measured from the wrong reference.
    """
    from dax_analog_explorer import session_builder as sb

    d = pd.Timestamp("2024-06-03")
    rows = []
    for hm, c in (((9, 0), 100.0), ((17, 25), 110.0), ((17, 30), 90.0)):
        t = pd.Timestamp(d.date(), tz=BER) + pd.Timedelta(hours=hm[0], minutes=hm[1])
        rows.append({"dt": t, "open": c, "high": c, "low": c, "close": c,
                     "adj_open": c, "adj_high": c, "adj_low": c, "adj_close": c,
                     "volume": 1.0, "con_id": pd.NA, "contract_month": pd.NA,
                     "instrument": "DAX5m", "source_file": "x"})
    s = sb.attach_sessions(pd.DataFrame(rows), Config())

    in_cash = s[s["is_cash"]]["dt"].dt.strftime("%H:%M").tolist()
    assert in_cash == ["09:00", "17:25"]          # 17:30 excluded
    # the closing price is the 17:25 bar (110), never the post-close 90
    assert float(in_cash and s[s["is_cash"]].sort_values("dt")["adj_close"].iloc[-1]) == 110.0
