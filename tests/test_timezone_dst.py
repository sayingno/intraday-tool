"""Europe/Berlin + Chicago DST handling."""
import pandas as pd
from dax_analog_explorer.data_loader import localize_to_berlin


def test_chicago_to_berlin_standard_offset_is_7h():
    # early March: both US and EU on winter/standard-ish -> 7h offset
    naive = pd.Series([pd.Timestamp("2019-03-08 02:00:00")])
    out, _ = localize_to_berlin(naive, "America/Chicago")
    assert out[0] == pd.Timestamp("2019-03-08 09:00", tz="Europe/Berlin")


def test_chicago_to_berlin_dst_mismatch_window_is_6h():
    # after US spring-forward (Mar 10) but before EU (Mar 31): offset drops to 6h
    naive = pd.Series([pd.Timestamp("2019-03-11 02:00:00")])
    out, _ = localize_to_berlin(naive, "America/Chicago")
    assert out[0] == pd.Timestamp("2019-03-11 08:00", tz="Europe/Berlin")


def test_berlin_localize_summer_and_winter():
    # FDAX timestamps are already Berlin wall-clock
    summer, _ = localize_to_berlin(pd.Series([pd.Timestamp("2025-07-01 09:00")]), "Europe/Berlin")
    winter, _ = localize_to_berlin(pd.Series([pd.Timestamp("2025-01-02 09:00")]), "Europe/Berlin")
    assert summer[0].utcoffset() == pd.Timedelta(hours=2)   # CEST
    assert winter[0].utcoffset() == pd.Timedelta(hours=1)   # CET


def test_result_timezone_is_berlin():
    out, _ = localize_to_berlin(pd.Series([pd.Timestamp("2020-06-01 03:00")]), "America/Chicago")
    assert str(out.tz) == "Europe/Berlin"
