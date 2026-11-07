import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import neighbour_reference as nb  # noqa: E402


def test_voting_limit():
    assert [nb.tolerated_failures(m) for m in (1, 2, 3, 4, 5)] == [0, 0, 1, 1, 2]


def test_alignment_recovers_a_reference_on_another_cadence():
    t5 = pd.date_range("2025-01-01", periods=12, freq="5min", tz="UTC")
    ref = pd.Series(1.0, index=pd.date_range("2025-01-01", periods=3, freq="20min", tz="UTC"))
    assert nb.align_series(ref, t5, 0).notna().sum() == 3
    assert nb.align_series(ref, t5, 14).notna().sum() == 9


def test_alignment_keeps_real_gaps():
    t = pd.date_range("2025-01-01", periods=7, freq="10min", tz="UTC")
    ref = pd.Series([1.0], index=t[:1])
    out = nb.align_series(ref, t, 14)
    assert out.notna().sum() == 2


def test_guard_abstains_below_quorum():
    ref = np.array([[1.0, 1.1, np.nan, np.nan, np.nan]])
    assert np.isnan(nb.guarded_median(ref, 8.0, quorum=3))[0]


def test_guard_uses_healthy_references():
    ref = np.array([[1.0, 1.1, 0.9, np.nan, np.nan]])
    assert abs(nb.guarded_median(ref, 8.0, quorum=3)[0] - 1.0) < 1e-9


def test_reference_health_removes_isolated_bias():
    ref = np.array([[0.0, 0.1, -0.1, 0.05, 15.0]])
    out = nb.reference_health_filter(ref, 8.0)
    assert np.isnan(out[0, 4]) and np.isfinite(out[0, :4]).all()


def test_majority_bias_is_not_solved():
    ref = np.array([[0.0, 0.1, 15.0, 15.1, 15.2]])
    assert nb.guarded_median(ref, 8.0, quorum=3)[0] > 8.0
