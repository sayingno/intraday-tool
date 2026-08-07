"""Observation-cutoff enforcement across path matching + outcomes."""
import numpy as np

from dax_analog_explorer.config import Config
from dax_analog_explorer import path_matching as pm
from dax_analog_explorer import outcome_engine as oe
from conftest import cash_frame, default_ref

CFG = Config()


def test_session_path_truncates_at_cutoff():
    bars = cash_frame(ohlc=[(100, 101, 99, 100 + i) for i in range(13)])  # mso 0..60
    ref = default_ref(cash_open=100.0)
    mso, val = pm.session_path(bars, cutoff_min=30, unit="points", ref=ref)
    assert mso.max() <= 30
    assert len(mso) == 7          # mso 0,5,...,30


def test_path_unaffected_by_future_bars():
    early = [(100, 101, 99, 100 + i) for i in range(7)]        # mso 0..30
    fut_a = [(106, 107, 105, 106)] * 6
    fut_b = [(106, 999, 1, 500)] * 6                            # wildly different future
    ref = default_ref(cash_open=100.0)
    _, va = pm.session_path(cash_frame(ohlc=early + fut_a), 30, "points", ref)
    _, vb = pm.session_path(cash_frame(ohlc=early + fut_b), 30, "points", ref)
    assert np.allclose(va, vb)


def test_observed_high_break_uses_only_pre_cutoff_high():
    # observed high (<=cutoff) is 102; after cutoff price makes 105 -> break True
    obs = [(100, 101, 99, 100), (100, 102, 100, 101), (101, 102, 100, 101),
           (101, 101.5, 100, 101), (101, 101.5, 100, 101), (101, 101.5, 100, 101),
           (101, 101.5, 100, 101)]
    after = [(101, 105, 100, 104)]
    o = oe.session_outcome(cash_frame(ohlc=obs + after), default_ref(cash_open=100.0), 30, CFG)
    assert o["break_observed_high"] is True
    # if nothing after cutoff exceeds the observed high, no break
    o2 = oe.session_outcome(cash_frame(ohlc=obs + [(101, 101.2, 100, 100.5)]),
                            default_ref(cash_open=100.0), 30, CFG)
    assert o2["break_observed_high"] is False


def test_empty_result_does_not_crash():
    """Filters that match nothing must return an empty result, not raise.

    Regression: build_outcomes([]) produced a DataFrame with NO columns, so
    search.run_search raised KeyError('session_date') on the merge instead of
    reporting "no matches".
    """
    import pandas as pd
    from dax_analog_explorer import outcome_engine as oe

    empty = oe.build_outcomes(pd.DataFrame(), pd.DataFrame(), 30, [], CFG)
    assert "session_date" in empty.columns
    assert len(empty) == 0
    # a merge on the key must now work rather than raise
    left = pd.DataFrame({"session_date": pd.to_datetime(["2024-01-02"])})
    assert len(left.merge(empty, on="session_date", how="left")) == 1
