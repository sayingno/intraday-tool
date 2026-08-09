"""Categorical context recognition — labels must mean what they say."""
import numpy as np
import pandas as pd

from dax_analog_explorer import context as ctx
from conftest import cash_frame

P = ctx.DEFAULT_PARAMS


def _open(ohlc, cash_open=100.0):
    return ctx.classify_opening(cash_frame(ohlc=ohlc), cash_open, P)


def test_rejection_up_needs_a_real_excursion_not_a_tick():
    """A one-tick poke above the open is a DRIVE_DOWN, not a REJECTION_UP.

    Regression: without the excursion floor, any bar that ticked past the open
    counted as a rejection and ~60% of all sessions were mislabelled.
    """
    # ran well above the open (up_ext ~0.6 of range) then closed on the lows
    rejection = _open([(100, 106, 99.5, 105), (105, 105, 101, 101), (101, 101.5, 99, 99.5)])
    assert rejection["opening"] == "REJECTION_UP"
    assert rejection["op_up_excursion"] >= P.excursion_frac

    # barely ticked above the open, then went straight down -> a drive, not a trap
    drive = _open([(100, 100.2, 97, 97.5), (97.5, 98, 95, 95.5), (95.5, 96, 94, 94.2)])
    assert drive["opening"] == "DRIVE_DOWN"
    assert drive["op_up_excursion"] < P.excursion_frac


def test_drive_up_and_rejection_down_are_mirror_images():
    drive_up = _open([(100, 103, 99.8, 102.5), (102.5, 105, 102, 104.5),
                      (104.5, 106, 104, 105.8)])
    assert drive_up["opening"] == "DRIVE_UP"

    rejection_down = _open([(100, 100.2, 94, 95), (95, 99, 94.5, 98.5),
                            (98.5, 101, 98, 100.8)])
    assert rejection_down["opening"] == "REJECTION_DOWN"
    assert rejection_down["op_down_excursion"] >= P.excursion_frac


def test_range_open_when_neither_side_commits():
    g = _open([(100, 101, 99, 100.2), (100.2, 101.5, 99.5, 100), (100, 100.8, 99.2, 99.9)])
    assert g["opening"] == "RANGE_OPEN"


def test_location_puts_ath_before_generic_range_labels():
    """AT_ATH must win over UPPER_RANGE / BREAKOUT_UP for the same session."""
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(np.linspace(100, 130, n))
    d = pd.DataFrame({
        "session_date": dates,
        "cash_open_adj": close, "cash_close_adj": close,
        "cash_high_adj": close + 1, "cash_low_adj": close - 1,
        "prior_ath": close.cummax().shift(1),
    })
    out = ctx.daily_context(d, P)
    # a relentless uptrend opens at/near its own prior high every day
    assert out["location"].iloc[-1] in ("AT_ATH", "NEAR_ATH")
    assert out["swing"].iloc[-1] in ("BULL", "STRONG_BULL")


def test_spec_filters_only_on_named_categories():
    t = pd.DataFrame({
        "location": ["AT_ATH", "MID_RANGE", "AT_ATH"],
        "swing": ["BULL", "BULL", "BEAR"],
        "opening": ["REJECTION_UP", "REJECTION_UP", "DRIVE_UP"],
    })
    spec = ctx.ContextSpec(location=("AT_ATH",), opening=("REJECTION_UP",))
    mask = ctx.apply_context(t, spec)
    assert mask.tolist() == [True, False, False]      # swing left unconstrained
    # a spec with nothing set keeps everything
    assert ctx.apply_context(t, ctx.ContextSpec()).all()


def test_funnel_is_monotonic():
    t = pd.DataFrame({
        "location": ["AT_ATH"] * 5 + ["MID_RANGE"] * 5,
        "swing": ["BULL"] * 3 + ["BEAR"] * 7,
        "opening": ["REJECTION_UP"] * 2 + ["DRIVE_UP"] * 8,
    })
    spec = ctx.ContextSpec(location=("AT_ATH",), swing=("BULL",), opening=("REJECTION_UP",))
    f = ctx.context_funnel(t, spec)
    surviving = f["surviving"].tolist()
    assert surviving == sorted(surviving, reverse=True)   # never grows
    assert surviving[0] == len(t)


def test_backtest_takes_the_stop_when_a_bar_holds_both_levels():
    """A 5m bar cannot prove which came first, so the stop must win."""
    from dax_analog_explorer import continuation as co
    from conftest import cash_frame

    # 6 opening bars trending up, then one bar spanning BOTH the target and the stop
    opening = [(100, 101, 99.5, 100.8), (100.8, 102, 100.5, 101.8),
               (101.8, 103, 101.5, 102.8), (102.8, 104, 102.5, 103.8),
               (103.8, 105, 103.5, 104.8), (104.8, 106, 104.5, 105.8)]
    # bar 7 engulfs target and stop alike; bar 8 just keeps the session going
    wild = [(105.8, 130, 90, 100), (100, 101, 99, 100)]
    bars = cash_frame(ohlc=opening + wild)
    r = co.simulate_day(bars, side=1, rules=co.TradeRules(target_R=2.0, cost_points=0))
    assert r["reason"] == "stop"
    assert r["R_multiple"] < 0


def test_entry_fills_on_the_bar_after_the_decision_never_inside_it():
    from dax_analog_explorer import continuation as co
    from conftest import cash_frame

    ohlc = [(100 + i, 101 + i, 99.5 + i, 100.8 + i) for i in range(6)]
    ohlc += [(200, 205, 195, 200), (200, 202, 198, 201)]   # bar 7 opens far away
    bars = cash_frame(ohlc=ohlc)
    r = co.simulate_day(bars, side=1, rules=co.TradeRules(cost_points=0))
    assert r["entry"] == 200.0               # the bar-7 open, not any bar-1..6 price
