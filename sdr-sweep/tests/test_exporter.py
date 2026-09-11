"""Exporter: R07 (unique series), R14 (per-sweep SNR), R15 (malformed input)."""
import io
import json

import pytest

from sdr_sweep import exporter


@pytest.fixture(autouse=True)
def reset_exporter():
    exporter._STATE = {"have": False}
    exporter._SWEEPS.clear()
    exporter._MARGIN = 6.0
    exporter._NBANDS = 8
    exporter._WINDOW = 60.0
    yield


def _samples(text):
    """Parse a Prometheus exposition into (identity, value) pairs (skip # lines)."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        ident, _, val = line.rpartition(" ")
        out.append((ident, float(val)))
    return out


def _value(text, name):
    for ident, val in _samples(text):
        if ident == name:
            return val
    raise KeyError(name)


def test_narrow_bands_emit_unique_series():
    # R07: 10 bins at 10 kHz over 8 bands used to emit several bands with the
    # same 0.1-MHz-rounded label -> duplicate metric identities.
    exporter.update({"f0": 915e6, "f1": 915.09e6, "bin": 1e4,
                     "db": [-90.0 + i for i in range(10)], "clip": 0.0})
    idents = [i for i, _ in _samples(exporter.render_metrics())]
    assert len(idents) == len(set(idents)), "every metric+label identity must be unique"
    # and all 8 bands are actually present (nothing silently merged away)
    band_peaks = [i for i in idents if i.startswith("sdr_sweep_band_peak_db{")]
    assert len(band_peaks) == 8


def test_snr_is_per_sweep_not_cross_window():
    # R14: two sweeps, each with the SAME peak-minus-(median)floor contrast of
    # 5 dB (db=[-100,-90] -> floor -95, peak -90). The old dashboard query
    # (peak_db_max - noise_floor_db_mean) crosses sweeps and reports 30 dB; the
    # new per-sweep sdr_sweep_snr_db_max reports the true 5 dB.
    exporter._NBANDS = 0
    exporter.update({"f0": 1e6, "f1": 2e6, "bin": 1e6, "db": [-100.0, -90.0], "clip": 0.0})
    exporter.update({"f0": 1e6, "f1": 2e6, "bin": 1e6, "db": [-50.0, -40.0], "clip": 0.0})
    text = exporter.render_metrics()
    assert _value(text, "sdr_sweep_snr_db_max") == pytest.approx(5.0)
    assert _value(text, "sdr_sweep_snr_db") == pytest.approx(5.0)    # latest sweep
    # the discredited cross-window form would have inflated this to 30 dB:
    old_form = _value(text, "sdr_sweep_peak_db_max") - _value(text, "sdr_sweep_noise_floor_db_mean")
    assert old_form == pytest.approx(30.0)


def test_read_stdin_skips_malformed_and_keeps_going():
    # R15: a `null` (and a bare list) between valid records must not stop the
    # consumer — both valid sweeps should still be counted.
    exporter._NBANDS = 0
    good = json.dumps({"f0": 1e6, "f1": 2e6, "bin": 1e6, "db": [-100.0, -90.0]})
    stream = "\n".join(["null", good, "[1, 2, 3]", "{bad json", good, ""])
    old_stdin = None
    import sys
    old_stdin, sys.stdin = sys.stdin, io.StringIO(stream)
    try:
        exporter.read_stdin()
    finally:
        sys.stdin = old_stdin
    assert exporter._STATE["sweeps"] == 2
