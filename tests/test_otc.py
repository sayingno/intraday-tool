"""Open Trend Continuation -- bar counting and the state machine.

The counting rules are the part that would silently corrupt every downstream
number if they were subtly wrong, so they are pinned on hand-built bars where the
intended label of each bar is obvious by eye.
"""
import numpy as np
import pandas as pd

from dax_analog_explorer import otc
from conftest import cash_frame

P = otc.OTCParams()


def _arrays(ohlc):
    a = np.array(ohlc, dtype=float)
    return a[:, 1], a[:, 2], a[:, 3], a[:, 0]     # h, l, c, o


# --------------------------------------------------------------------------- #
# bar counting
# --------------------------------------------------------------------------- #
def test_two_legged_pullback_labels_h1_then_h2():
    #  bars 0-3 rally (the leg), 4-5 pull back, 6 = H1, 7 pulls back again,
    #  8 = H2.  No bar makes a new high, so the count never resets.
    ohlc = [(100, 102, 99, 101.5),      # 0
            (101.5, 104, 101, 103.5),   # 1
            (103.5, 106, 103, 105.5),   # 2
            (105.5, 108, 105, 107.5),   # 3  leg extreme 108
            (107.5, 107, 104, 104.5),   # 4  lower high -> correction opens
            (104.5, 106, 103, 103.5),   # 5  lower high still
            (103.5, 106.5, 103, 106),   # 6  higher high -> H1
            (106, 105.5, 102, 102.5),   # 7  lower high -> second leg down
            (102.5, 106, 102, 105.5)]   # 8  higher high -> H2
    h, l, c, o = _arrays(ohlc)
    leg = otc.detect_leg(h, l, c, o, len(ohlc) - 1)
    assert leg is not None and leg.side == 1
    sig = otc.bar_count(h, l, leg, len(ohlc) - 1)
    assert [(s.label, s.signal_i) for s in sig] == [("H1", 6), ("H2", 8)]


def test_bear_mirror_labels_l1_then_l2():
    ohlc = [(100, 101, 98, 98.5),
            (98.5, 99, 96, 96.5),
            (96.5, 97, 94, 94.5),
            (94.5, 95, 92, 92.5),       # leg extreme 92
            (92.5, 96, 93, 95.5),       # higher low -> correction opens
            (95.5, 97, 94, 96.5),       # higher low still
            (96.5, 97, 93.5, 94),       # lower low -> L1
            (94, 98, 95.5, 97.5),       # higher low -> second leg up
            (97.5, 98, 94.8, 95)]       # lower low -> L2
    h, l, c, o = _arrays(ohlc)
    leg = otc.detect_leg(h, l, c, o, len(ohlc) - 1)
    assert leg is not None and leg.side == -1
    sig = otc.bar_count(h, l, leg, len(ohlc) - 1)
    assert [(s.label, s.signal_i) for s in sig] == [("L1", 6), ("L2", 8)]


def test_count_resets_when_the_trend_makes_a_new_extreme():
    #  H1 at bar 6, then bar 7 makes a NEW HIGH -> the count resets, so the
    #  signal after the next correction is H1 again, not H2.
    ohlc = [(100, 102, 99, 101.5),
            (101.5, 104, 101, 103.5),
            (103.5, 106, 103, 105.5),
            (105.5, 108, 105, 107.5),
            (107.5, 107, 104, 104.5),   # 4 lower high
            (104.5, 106, 103, 103.5),   # 5 lower high
            (103.5, 106.5, 103, 106),   # 6 higher high -> H1
            (106, 110, 105, 109.5),     # 7 NEW HIGH -> reset
            (109.5, 109, 106, 106.5),   # 8 lower high
            (106.5, 109.4, 106, 108)]   # 9 higher high -> H1 again
    h, l, c, o = _arrays(ohlc)
    leg = otc.detect_leg(h, l, c, o, len(ohlc) - 1)
    sig = otc.bar_count(h, l, leg, len(ohlc) - 1)
    assert [(s.label, s.signal_i) for s in sig] == [("H1", 6), ("H1", 9)]


def test_higher_high_without_a_preceding_lower_high_is_not_a_signal():
    # a clean staircase up: every bar has a higher high, no correction ever opens
    ohlc = [(100 + i, 102 + i, 99 + i, 101.5 + i) for i in range(6)]
    h, l, c, o = _arrays(ohlc)
    leg = otc.detect_leg(h, l, c, o, len(ohlc) - 1)
    assert leg is not None
    assert otc.bar_count(h, l, leg, len(ohlc) - 1) == []


# --------------------------------------------------------------------------- #
# the rules
# --------------------------------------------------------------------------- #
def test_one_big_bar_fails_follow_through():
    # a single huge bar then nothing extends it -> rule 2 rejects the leg
    ohlc = [(100, 120, 99, 119), (119, 119.5, 117, 118),
            (118, 119, 117, 118.5), (118.5, 119, 117.5, 118)]
    h, l, c, o = _arrays(ohlc)
    assert otc.detect_leg(h, l, c, o, len(ohlc) - 1) is None


def test_vertical_climax_stays_developing_and_never_triggers():
    """Rule 3 is enforced by rule 4: no pullback means no trigger."""
    bars = cash_frame(ohlc=[(100 + 3 * i, 103 + 3 * i, 99.5 + 3 * i, 102.5 + 3 * i)
                            for i in range(8)])
    res = otc.evaluate_session(bars, {"PDH": 200.0}, P)
    assert res.state == otc.DEVELOPING
    assert res.signals == []


def test_pullback_through_the_leg_origin_invalidates():
    ohlc = [(100, 102, 99, 101.5), (101.5, 104, 101, 103.5),
            (103.5, 106, 103, 105.5), (105.5, 107, 104, 104.5),
            (104.5, 105, 96, 96.5),          # collapses below the origin (100)
            (96.5, 99, 95, 98)]
    res = otc.evaluate_session(cash_frame(ohlc=ohlc), {"PDH": 200.0}, P)
    assert res.state == otc.INVALID


def test_origin_held_ignores_bar_one_and_catches_later_breaks():
    """Bar 1's own low IS the origin, so it can never be the violation."""
    h = np.array([102., 104., 106., 107.]); l = np.array([99., 101., 103., 104.])
    leg = otc.Leg(side=1, origin_px=99.0, origin_i=0, extreme_px=107.0,
                  extreme_i=3, hh_hl=3, followthrough_bars=3)
    assert otc.origin_held(h, l, leg, 3) is True     # nothing later broke 99
    broke = np.array([99., 101., 98.5, 104.])        # bar 2 trades under it
    assert otc.origin_held(h, broke, leg, 3) is False


# --------------------------------------------------------------------------- #
# targets, scope, discipline
# --------------------------------------------------------------------------- #
def test_target_is_the_nearest_untouched_level_ahead():
    lv = {"PDH": 120.0, "ONH": 111.0, "PDL": 90.0}
    assert otc.next_structural_target(105.0, 1, lv) == ("ONH", 111.0)
    assert otc.next_structural_target(105.0, -1, lv) == ("PDL", 90.0)
    # nothing ahead -> no target rather than a fabricated one
    assert otc.next_structural_target(130.0, 1, {"PDH": 120.0}) is None


def test_no_signal_after_the_deadline():
    ohlc = [(100, 102, 99, 101.5), (101.5, 104, 101, 103.5),
            (103.5, 106, 103, 105.5), (105.5, 108, 105, 107.5),
            (107.5, 107, 104, 104.5), (104.5, 106, 103, 103.5),
            (103.5, 106.5, 103, 106)]               # the H1 bar sits at mso 30
    bars = cash_frame(ohlc=ohlc)
    assert otc.evaluate_session(bars, {"PDH": 200.0},
                                otc.OTCParams(trigger_deadline_min=30)).signals
    assert otc.evaluate_session(bars, {"PDH": 200.0},
                                otc.OTCParams(trigger_deadline_min=25)).signals == []


def test_third_signal_is_logged_but_not_taken():
    ohlc = [(100, 102, 99, 101.5), (101.5, 104, 101, 103.5),
            (103.5, 106, 103, 105.5), (105.5, 108, 105, 107.5)]
    for _ in range(5):                              # enough separate corrections
        ohlc += [(107.5, 107, 104, 104.5), (104.5, 106.5, 103.5, 106)]
    res = otc.evaluate_session(cash_frame(ohlc=ohlc), {"PDH": 200.0}, P)
    assert len(res.signals) >= 3
    assert [s.taken for s in res.signals][:3] == [True, True, False]
    assert res.signals[2].reason == "max_entries reached"


def test_live_status_reports_the_state_it_reached():
    bars = cash_frame(ohlc=[(100 + 3 * i, 103 + 3 * i, 99.5 + 3 * i, 102.5 + 3 * i)
                            for i in range(8)])
    txt = otc.live_status(otc.evaluate_session(bars, {"PDH": 200.0}, P), P)
    assert "DEVELOPING" in txt and "WAIT" in txt


def test_entry_is_cancelled_when_the_stop_level_breaks_before_the_fill():
    """A pullback that runs through its own stop kills the order.

    Regression: the engine used to leave the entry working, fill it much later,
    and book the result against a stop that had already been violated -- 1404 of
    4735 "trades" were setups no trader would still have had on.
    """
    from dax_analog_explorer import otc, otc_report as orp
    from conftest import cash_frame

    sig = otc.OTCSignal(label="H1", side=1, signal_mso=25, entry_px=110.0,
                        stop_px=95.0, target_name="PDH", target_px=120.0,
                        rr=0.67, taken=True)
    # after the signal: price collapses through 95 first, then rallies past 110
    after = [(105, 106, 90, 92), (92, 100, 91, 99), (99, 115, 98, 114)]
    bars = cash_frame(ohlc=[(100, 101, 99, 100)] * 6 + after)
    r = orp.simulate_signal(bars, sig, otc.OTCParams())
    assert r["status"] == "CANCELLED"

    # the mirror: the entry fills before anything reaches the stop -> a real trade
    ok = [(105, 112, 104, 111), (111, 121, 110, 120)]
    bars2 = cash_frame(ohlc=[(100, 101, 99, 100)] * 6 + ok)
    r2 = orp.simulate_signal(bars2, sig, otc.OTCParams())
    assert r2["status"] == "TRADED" and r2["reason"] == "target"


def test_a_trigger_that_already_fired_is_not_hidden_by_a_later_quiet_bar():
    """State is "what do I do now"; reached is "did this day ever produce it".

    Regression: the walk set the state from the LAST bar, so a session that
    counted an H1 at 10:00 and then simply ran reported DEVELOPING/WAIT at the
    deadline and never mentioned the trigger -- and --profile, counting the same
    field, answered "did a trigger fire on the very final bar" instead of "how
    many days reach state C".
    """
    ohlc = [(100, 102, 99, 101.5), (101.5, 104, 101, 103.5),
            (103.5, 106, 103, 105.5), (105.5, 108, 105, 107.5),
            (107.5, 107, 104, 104.5),        # lower high -> the correction opens
            (104.5, 106, 103, 103.5),        # still correcting
            (103.5, 106.5, 103, 106)]        # H1: higher high, still under the leg high
    quiet = [(106, 106.2, 105.8, 106.0)] * 3   # nothing new happens afterwards
    res = otc.evaluate_session(cash_frame(ohlc=ohlc + quiet), {"PDH": 200.0}, P)

    assert [s.label for s in res.signals] == ["H1"]
    assert res.state == otc.DEVELOPING        # the last bar counted nothing
    assert res.reached == otc.ACTIVE          # but the day did produce the setup

    txt = otc.live_status(res, P)
    assert "H1 at 09:30" in txt               # the trigger and its clock time
    assert "NO NEW ENTRY" in txt              # not a bare "WAIT"


def test_clock_maps_minutes_after_the_open_to_berlin_time():
    assert otc.clock(0) == "09:00"
    assert otc.clock(60) == "10:00"
    assert otc.clock(90) == "10:30"
