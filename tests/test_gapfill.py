"""Gap-fill logic (full vs half; touching the open is NOT a full fill)."""
import numpy as np

from dax_analog_explorer.config import Config
from dax_analog_explorer import outcome_engine as oe
from conftest import cash_frame, default_ref

CFG = Config()


def _outcome(after_lows, cutoff=30):
    # 7 observed bars (mso 0..30), then "after" bars whose lows we control
    obs = [(100, 100.5, 99.8, 100.2)] * 7
    after = [(100.2, 100.3, lo, 100.0) for lo in after_lows]
    bars = cash_frame(ohlc=obs + after)
    ref = default_ref(cash_open=100.0, prev_close=98.0)   # gap up +2
    return oe.session_outcome(bars, ref, cutoff, CFG)


def test_full_gap_fill_when_price_reaches_prev_close():
    o = _outcome(after_lows=[99.5, 98.9, 97.5])   # dips to 97.5 <= 98 prev close
    assert o["full_gap_fill"] is True
    assert o["half_gap_fill"] is True


def test_half_but_not_full():
    o = _outcome(after_lows=[99.4, 98.7])          # reaches 98.7 (<=99 half) not 98
    assert o["half_gap_fill"] is True
    assert o["full_gap_fill"] is False


def test_touching_open_is_not_a_full_fill():
    # price returns to the open (100) and just below, but never to prev close (98)
    o = _outcome(after_lows=[100.0, 99.6, 99.4])
    assert o["full_gap_fill"] is False
    assert o["half_gap_fill"] is False


def test_gap_down_full_fill_direction():
    # gap DOWN: open below prev close; fill means price rises back to prev close
    obs = [(100, 100.2, 99.8, 100.0)] * 7
    after = [(100.0, hi, 99.9, 100.1) for hi in [101.0, 102.1]]  # rises to 102.1
    bars = cash_frame(ohlc=obs + after)
    ref = default_ref(cash_open=100.0, prev_close=102.0)         # gap down -2
    o = oe.session_outcome(bars, ref, 30, CFG)
    assert o["full_gap_fill"] is True
