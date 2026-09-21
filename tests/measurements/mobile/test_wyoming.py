"""Tests for slv.measurements.mobile.wyoming."""

import numpy as np
import pandas as pd

from slv.measurements.mobile.wyoming import calculate_enhancements


def test_enhancement_baseline_window_is_in_hours():
    idx = pd.date_range(
        "2024-08-05 12:00", periods=4 * 3600, freq="s", tz="America/Denver"
    )
    ch4 = np.full(len(idx), 2000.0)
    ch4[7200:7260] += 500.0  # a one-minute plume
    data = pd.DataFrame({"CH4 (ppb)": ch4, "C2H6 (ppb)": 5.0}, index=idx)
    out = calculate_enhancements(data.copy())
    assert out.CH4_ex.iloc[7230] > 400 and abs(out.CH4_ex.iloc[0]) < 1
    # a bare number is hours, the same as the default "1h"
    same = calculate_enhancements(data.copy(), window=1)
    pd.testing.assert_series_equal(same.CH4_ex, out.CH4_ex)
