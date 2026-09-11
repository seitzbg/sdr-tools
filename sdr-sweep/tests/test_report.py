"""report: R01 (persistent carriers), R13 (missing obs), R12 (mixed grid),
R16 (sub-second timestamps), R05 (bin-centre round trip)."""
import numpy as np
import pytest

from sdr_sweep import report, sweep


# --- R01: a persistent carrier must not vanish into its own baseline ---------

def test_persistent_carrier_is_detected():
    P = np.full((100, 20), -100.0)
    P[:, 10] = -30.0                       # one bin held high in every sweep
    times = np.arange(100, dtype=float)
    freqs = np.arange(20, dtype=float) * 1e4 + 900e6
    a = report.analyse(times, freqs, P, margin=6.0, floor_pct=25.0)
    assert a["occupancy"][10] > 0.9, "steady carrier should read as fully occupied"
    assert any(e["kind"] == "persistent" for e in a["emitters"])
    # a genuinely quiet bin stays quiet
    assert a["occupancy"][0] < 0.05


def test_intermittent_carrier_still_detected():
    # Guard against regressing the previously-correct intermittent case.
    P = np.full((100, 20), -100.0)
    P[::4, 10] = -40.0                      # active in 25% of sweeps
    times = np.arange(100, dtype=float)
    freqs = np.arange(20, dtype=float) * 1e4 + 900e6
    a = report.analyse(times, freqs, P, margin=6.0, floor_pct=25.0)
    assert a["occupancy"][10] == pytest.approx(0.25, abs=0.02)
    assert len(a["emitters"]) == 1


# --- R13: missing observations excluded from occupancy, not counted inactive -

def test_missing_observations_not_counted_inactive():
    P = np.array([[-100.0, -100.0],
                  [-80.0, -100.0],
                  [np.nan, -100.0],
                  [np.nan, -100.0]])
    times = np.arange(4, dtype=float)
    freqs = np.array([900e6, 900.01e6])
    a = report.analyse(times, freqs, P, margin=6.0, floor_pct=25.0)
    # bin 0 was observed twice, active once -> 50%, not 1/4 = 25%.
    assert a["occupancy"][0] == pytest.approx(0.5)


# --- R12: appending a different bin config is rejected, not mis-binned -------

def test_mixed_bin_config_rejected(tmp_path):
    csv = tmp_path / "mixed.csv"
    csv.write_text(
        "2026-01-01, 00:00:00, 0, 40000, 10000.00, 100, -90, -90, -90, -90\n"
        "2026-01-01, 00:00:01, 0, 40000, 20000.00, 100, -90, -90\n")
    with pytest.raises(SystemExit):
        report.load(str(csv))


# --- R16: sub-second timestamps preserve the timing of fast sweeps ----------

def test_subsecond_timestamps_preserved(tmp_path):
    csv = tmp_path / "fast.csv"
    lines = []
    for frac in ("100", "400", "800"):
        lines.append(f"2026-01-01, 00:00:00.{frac}, 0, 20000, 10000.00, 100, -90, -90")
    csv.write_text("\n".join(lines) + "\n")
    times, freqs, P = report.load(str(csv))
    assert len(times) == 3
    assert (times[-1] - times[0]) == pytest.approx(0.7, abs=1e-3)


def test_legacy_whole_second_timestamps_still_load(tmp_path):
    csv = tmp_path / "legacy.csv"
    csv.write_text("2026-01-01, 00:00:00, 0, 20000, 10000.00, 100, -90, -90\n")
    times, freqs, P = report.load(str(csv))
    assert len(times) == 1


# --- R05: CSV bin edges round-trip back to the true FFT bin centres ----------

def test_csv_roundtrip_recovers_true_bin_centers(tmp_path):
    center, keep, bin_hz = 100.5e6, 8, 10e3
    centers = sweep.hop_bin_centers(center, keep, bin_hz)
    low = centers[0] - bin_hz / 2.0        # exactly how the capture engine writes edges
    high = centers[-1] + bin_hz / 2.0
    row = ["2026-01-01", "00:00:00.000", f"{low:.0f}", f"{high:.0f}",
           f"{bin_hz:.2f}", "100"] + [f"{-90.0:.2f}"] * keep
    csv = tmp_path / "one.csv"
    csv.write_text(", ".join(row) + "\n")
    _, freqs, _ = report.load(str(csv))
    assert np.allclose(freqs, centers, atol=1.0), "no half-bin offset on the round trip"
