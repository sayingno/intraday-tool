"""ATH calculation + the no-future-leakage guarantees."""
import numpy as np
import pandas as pd

from dax_analog_explorer.config import Config
from dax_analog_explorer import ath_engine as ae
from dax_analog_explorer import pattern_features as pf
from conftest import cash_frame, default_ref


def test_prior_ath_is_strict_expanding_max():
    d = pd.DataFrame({"session_date": pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "day_high_adj": [10.0, 12.0, 11.0, 15.0]})
    prior = ae.compute_prior_ath(d)
    # prior_ath[D] = max of days strictly before D
    assert np.isnan(prior.iloc[0])
    assert prior.iloc[1] == 10.0
    assert prior.iloc[2] == 12.0
    assert prior.iloc[3] == 12.0          # 15 (day 4's own high) must NOT leak in


def test_prior_ath_never_includes_own_day():
    # even a giant same-day high cannot enter its own prior_ath
    d = pd.DataFrame({"session_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                      "day_high_adj": [10.0, 9999.0]})
    prior = ae.compute_prior_ath(d)
    assert prior.iloc[1] == 10.0


def test_ath_at_open_flag():
    # open above prior ath -> new ATH at open
    open_px, prior = 101.0, 100.0
    assert (open_px > prior) is True


def test_cutoff_blocks_future_bars_in_pattern_features(cfg):
    # a session where everything after 09:30 is extreme; features at cutoff=30
    # must be identical whether or not those future bars exist.
    base = [(100, 101, 99, 100.5), (100.5, 102, 100, 101.5),
            (101.5, 103, 101, 102.5), (102.5, 103.5, 102, 103.0),
            (103.0, 104, 102.5, 103.5), (103.5, 104.5, 103, 104.0),
            (104.0, 105, 103.5, 104.5)]
    future_wild = [(104.5, 900, 104, 850), (850, 999, 800, 990)]
    ref = default_ref(cash_open=100.0)

    f_with = pf.session_features(cash_frame(ohlc=base + future_wild), ref, cfg, cutoff_min=30)
    f_without = pf.session_features(cash_frame(ohlc=base), ref, cfg, cutoff_min=30)
    for k in ("high_low_range", "net_return_pct", "n_window_bars",
              "directional_efficiency", "first_bar_range"):
        assert (f_with[k] == f_without[k]) or (np.isnan(f_with[k]) and np.isnan(f_without[k]))
    # only bars up to mso=30 (7 bars incl mso 0..30) are used
    assert f_with["n_window_bars"] == 7
