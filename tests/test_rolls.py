"""Contract-roll / back-adjustment handling."""
import numpy as np
import pandas as pd

from dax_analog_explorer.config import Config
from dax_analog_explorer import session_builder as sb
from conftest import fdax_synthetic


def test_seam_gaps_detected():
    fd = fdax_synthetic()
    ri = sb.compute_roll_adjustment(fd, Config())
    # two seams, each ~+10 points (100->110, 110->120)
    assert len(ri.seams) == 2
    assert all(abs(s["seam_gap_pts"] - 10.0) < 1e-6 for s in ri.seams)


def test_difference_method_makes_series_continuous():
    cfg = Config(); cfg.roll_adjust_method = "difference"
    fd = fdax_synthetic()
    ri = sb.compute_roll_adjustment(fd, cfg)
    # newest contract anchored raw
    newest = fd.groupby("con_id")["contract_month"].first().idxmax()
    assert ri.offsets[newest] == 0.0
    assert ri.verification["newest_contract_anchored_raw"] is True
    # after additive adjustment the three flat contracts line up at one level
    raw = pd.Series([100.0, 110.0, 120.0])
    cids = pd.Series([111, 222, 333])
    is_dax = pd.Series([False, False, False])
    adj = ri.adjust(raw, cids, is_dax)
    assert np.allclose(adj.to_numpy(), 120.0)
    assert ri.verification["seam_residual_pct_max"] < 1e-6


def test_carry_method_anchors_newest_and_preserves_returns():
    cfg = Config(); cfg.roll_adjust_method = "carry"
    ri = sb.compute_roll_adjustment(fdax_synthetic(), cfg)
    newest = 333
    assert abs(ri.factors[newest] - 1.0) < 1e-12
    assert ri.multiplicative is True
    assert ri.verification["preserves_pct_returns"] is True
    # carry residual is the (large) real move left in place, not ~0
    assert ri.verification["seam_residual_pct_max"] > 1.0


def test_none_method_is_identity():
    cfg = Config(); cfg.roll_adjust_method = "none"; cfg.roll_adjust_enabled = False
    ri = sb.compute_roll_adjustment(fdax_synthetic(), cfg)
    raw = pd.Series([100.0, 110.0, 120.0])
    adj = ri.adjust(raw, pd.Series([111, 222, 333]), pd.Series([False] * 3))
    assert np.allclose(adj.to_numpy(), raw.to_numpy())
