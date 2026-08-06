"""Pure price-action geometry + the conditional-probability gate."""
import numpy as np
import pandas as pd

from dax_analog_explorer.config import Config
from dax_analog_explorer import price_action as pa
from conftest import cash_frame


def _geo(ohlc, cash_open=100.0, window=15):
    return pa.opening_geometry(cash_frame(ohlc=ohlc), cash_open, window)


def test_close_location_is_scaled_by_the_window_range_only():
    # window range 96..104; close 97 -> location (97-96)/8 = 0.125, no ATR anywhere
    g = _geo([(100, 104, 99, 103), (103, 103, 96, 97), (97, 98, 96, 97)])
    assert g["win_high"] == 104 and g["win_low"] == 96
    assert abs(g["close_location"] - (97 - 96) / 8) < 1e-9


def test_selling_pressure_geometry_flags():
    # bar1 pokes above the open then the window closes below it, on its lows
    g = _geo([(100, 103, 99, 100), (100, 101, 96, 97), (97, 98, 95, 95.5)])
    assert g["poked_above_open"] is True
    assert g["close_below_open"] is True
    assert g["spike_up_reject"] is True          # traded above the open, closed below
    assert g["monotonic_lower_highs"] is True    # 103 > 101 > 98
    assert g["close_below_prior_low"] is True    # bar2 closed 97 < bar1 low 99
    assert g["n_bear_bars"] == 2


def test_open_is_high_when_no_bar_trades_above_the_open():
    g = _geo([(100, 100, 98, 98.5), (98.5, 99, 97, 97.5), (97.5, 98, 96, 96.5)])
    assert g["open_is_high"] is True
    assert g["poked_above_open"] is False


def test_pressure_gate_respects_each_parameter():
    ohlc = [(100, 103, 99, 100), (100, 101, 96, 97), (97, 98, 95, 95.5)]
    g = _geo(ohlc)
    base = dict(ath_tolerance_pct=None, require_open_above_pdh=False,
                gap_min_pct=None, gap_max_pct=None, on_location_min=None)

    assert pa._passes_pressure(g, pa.PriceActionSpec(**base, close_location_max=0.5))
    # demanding a close in the bottom 1% of the range must reject this window
    assert not pa._passes_pressure(g, pa.PriceActionSpec(**base, close_location_max=0.01))
    # monotonic highs hold here, ≥4 bear bars cannot (only 3 bars exist)
    assert pa._passes_pressure(g, pa.PriceActionSpec(**base, require_monotonic_highs=True))
    assert not pa._passes_pressure(g, pa.PriceActionSpec(**base, min_directional_bars=4))


def test_conditional_gate_excludes_levels_already_tested():
    """A level touched before the decision bar leaves its own denominator."""
    m = pd.DataFrame({
        "session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        # PDH: day1 already tested pre-decision; days 2-3 untested, day2 reaches it
        "pre_PDH": [True, False, False],
        "morn_PDH": [True, True, False],
        "aft_PDH": [True, True, False],
    })
    rep = pa.conditional_report(m).set_index("level")
    row = rep.loc["PDH"]
    assert row["tested_before_decision"] == 1
    assert row["untested_at_decision"] == 2      # day1 removed from the denominator
    assert row["reached_in_morning"] == 1
    assert row["p_morning_%"] == 50.0            # 1 of 2, not 2 of 3


def test_spec_describe_lists_conditions_in_order():
    s = pa.PriceActionSpec()
    d = s.describe()
    assert any("prior ATH" in x for x in d)
    assert any("PDH" in x for x in d)
    assert any("gap" in x for x in d)
    assert s.decision_min == 30                  # bar 6 on a 5m chart = 09:30
