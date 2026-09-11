"""TUI helpers: R08 (narrow-spectrum crash) and R17 (peak-hold decay)."""
import numpy as np

from sdr_sweep import tui


def test_pool_downsamples_to_width():
    pooled = tui.pool(np.arange(1000.0), 120)
    assert pooled.shape == (120,)


def test_pool_expands_narrow_spectrum_to_width():
    # R08: fewer bins than display columns must still fill the width, so the
    # render loop's range(cols) never indexes past the pooled array.
    for n in (1, 5, 10, 39):
        pooled = tui.pool(np.arange(float(n)), 40)
        assert pooled.shape == (40,), f"n={n} -> {pooled.shape}"
        # nearest-neighbour upsample preserves the value range
        assert pooled.min() == 0.0 and pooled.max() == n - 1


def test_pool_max_pool_keeps_narrow_peak():
    db = np.full(1000, -100.0)
    db[503] = -10.0                       # a one-bin spike
    assert tui.pool(db, 100).max() == -10.0


def _peak_hold_step(peak_hold, pdb):
    # Mirrors tui.render's decay line exactly.
    return np.maximum(peak_hold - 0.5, pdb)


def test_peak_hold_decays_downward_for_negative_db():
    # R17: a held peak must fade toward the current (weaker) level, not grow
    # toward 0 dB as `* 0.995` did on negative values.
    held = np.array([-40.0])
    weaker = np.array([-80.0])
    h1 = _peak_hold_step(held, weaker)
    assert h1[0] < held[0], "held peak must decay downward"
    h2 = _peak_hold_step(h1, weaker)
    assert h2[0] < h1[0]
    # and it never runs away toward zero
    for _ in range(1000):
        held = _peak_hold_step(held, weaker)
    assert held[0] <= -79.9
