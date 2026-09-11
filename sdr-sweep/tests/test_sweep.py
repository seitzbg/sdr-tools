"""sweep engine: R09 (validation), R10 (run timing), R05/R11/R16/R06 via a
fake UHD streamer feeding a known synthetic tone."""
import json
import sys
import types

import numpy as np
import pytest

from sdr_sweep import sweep


# --- R09: degenerate scan parameters are rejected, not divided by zero -------

@pytest.mark.parametrize("kw", [
    {"bin_hz": 0.0},
    {"hop_bw": 0.0},
    {"crop": 1.0},
    {"crop": 1.5},
    {"crop": -0.1},
    {"frames": 0},
    {"bin_hz": 30e6},        # bin >= hop_bw -> fewer than 4 FFT bins
])
def test_validate_scan_rejects_degenerate(kw):
    args = {"hop_bw": 20e6, "bin_hz": 10e3, "crop": 0.2, "frames": 16}
    args.update(kw)
    with pytest.raises(ValueError):
        sweep.validate_scan(**args)


def test_validate_scan_accepts_normal():
    sweep.validate_scan(20e6, 10e3, 0.2, 16)   # no raise


def test_bins_in_range_counts_only_in_range():
    # zero when the hop's bins fall entirely outside the window...
    assert sweep.bins_in_range([1000.0], 4, 10.0, 5000.0, 6000.0) == 0
    # ...and positive for a normal covering plan. (Note: with the R05 grid
    # fixed, a parked capture always has a bin at the midpoint of [start,stop],
    # so the guard's real job is the degenerate keep/plan case; the ZeroDivision
    # inputs are caught earlier by validate_scan.)
    _, _, bin_hz, keep, centers, _ = sweep.build_hop_plan(915e6, 917e6, 20e6, 10e3, 0.2)
    assert sweep.bins_in_range(centers, keep, bin_hz, 915e6, 917e6) > 0


def test_hop_bin_centers_have_no_half_bin_offset():
    center, keep, bin_hz = 100.5e6, 128, 10e3
    fc = sweep.hop_bin_centers(center, keep, bin_hz)
    # the true FFT grid puts a bin exactly on the tuning centre (index keep/2)
    assert fc[keep // 2] == pytest.approx(center)
    assert fc[0] == pytest.approx(center - keep / 2 * bin_hz)


# --- R10: a run never overruns its --duration because of --interval ----------

class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_interval_does_not_overrun_duration():
    clk = _FakeClock()
    sweeps = []
    for n in sweep.sweep_schedule(clk, count=0, duration=1.0, interval=60.0):
        sweeps.append(n)
        clk.t += 0.1                       # simulate a 0.1 s sweep
        if len(sweeps) > 5:                # safety net so a bug can't hang the test
            break
    assert len(sweeps) == 1, "only one sweep fits in a 1 s run with a 60 s interval"
    assert clk.t <= 1.05, f"run should end near the 1 s deadline, not at 60 s (got {clk.t})"


def test_schedule_honours_count():
    clk = _FakeClock()
    got = [n for n in sweep.sweep_schedule(clk, count=3, duration=0.0, interval=0.0)]
    assert got == [0, 1, 2]


def test_schedule_unlimited_stops_on_count_under_duration():
    clk = _FakeClock()
    got = []
    for n in sweep.sweep_schedule(clk, count=3, duration=100.0, interval=0.0):
        got.append(n)
        clk.t += 0.1
    assert got == [0, 1, 2]


# --- Fake UHD: a synthetic +100 kHz tone through the real capture path -------

def _fake_uhd(f_tone, actual_rate, width=8192):
    class FakeUSRP:
        def __init__(self):
            self.center = 0.0
        def set_rx_antenna(self, *a):
            pass
        def set_rx_rate(self, *a):
            pass
        def get_rx_rate(self, *a):
            return actual_rate
        def set_rx_gain(self, *a):
            pass
        def set_rx_dc_offset(self, *a):
            pass
        def set_rx_iq_balance(self, *a):
            pass
        def set_rx_freq(self, tune, ch):
            self.center = tune.freq
        def get_rx_stream(self, st):
            return self._streamer

    class FakeStreamer:
        def __init__(self, usrp):
            self.usrp = usrp
            self.n0 = 0
        def get_max_num_samps(self):
            return width
        def issue_stream_cmd(self, cmd):
            self.n0 = 0
        def recv(self, recv_buf, md, timeout):
            w = recv_buf.shape[1]
            idx = np.arange(self.n0, self.n0 + w)
            sig = np.exp(2j * np.pi * (f_tone - self.usrp.center) / actual_rate * idx)
            recv_buf[0, :w] = sig.astype(np.complex64)
            self.n0 += w
            return w

    usrp = FakeUSRP()
    usrp._streamer = FakeStreamer(usrp)

    class TuneRequest:
        def __init__(self, freq):
            self.freq = freq

    class StreamCMD:
        def __init__(self, mode):
            self.mode = mode
            self.num_samps = 0
            self.stream_now = False

    class RXMetadata:
        def strerror(self):
            return ""

    class StreamArgs:
        def __init__(self, *a):
            self.channels = [0]

    mod = types.ModuleType("uhd")
    mod.types = types.SimpleNamespace(
        TuneRequest=TuneRequest, StreamCMD=StreamCMD, RXMetadata=RXMetadata,
        StreamMode=types.SimpleNamespace(num_done="num_done"))
    mod.usrp = types.SimpleNamespace(MultiUSRP=lambda: usrp, StreamArgs=StreamArgs)
    return mod


def _run_capture(monkeypatch, capsys, argv, f_tone, actual_rate):
    monkeypatch.setitem(sys.modules, "uhd", _fake_uhd(f_tone, actual_rate))
    monkeypatch.setattr(sys, "argv", argv)
    sweep.main()
    return capsys.readouterr().out


BASE = ["sdr-sweep", "--start", "100.0M", "--stop", "101.0M",
        "--hop-bw", "1.28M", "--bin", "10k", "--count", "1", "--quiet"]


def test_stream_tone_lands_on_correct_frequency(monkeypatch, capsys):
    # R05: a +100 kHz tone (centre 100.5 MHz) must peak at 100.600 MHz exactly,
    # not 100.605 MHz.
    out = _run_capture(monkeypatch, capsys, BASE + ["--stream"], f_tone=100.6e6, actual_rate=1.28e6)
    obj = json.loads(out.strip().splitlines()[-1])
    db = np.array(obj["db"])
    peak_f = obj["f0"] + int(np.argmax(db)) * obj["bin"]
    assert peak_f == pytest.approx(100.6e6, abs=5e3)
    # R11: nothing beyond the requested range leaks into the stream
    assert obj["f1"] <= 101.0e6 + 1.0


def test_csv_is_trimmed_and_has_subsecond_time(monkeypatch, capsys, tmp_path):
    # R11 + R16 on the CSV path, then a full round trip back through the reader.
    from sdr_sweep import report
    csv = tmp_path / "cap.csv"
    _run_capture(monkeypatch, capsys, BASE + ["-o", str(csv)], f_tone=100.6e6, actual_rate=1.28e6)
    text = csv.read_text().strip()
    assert text, "a CSV row should have been written"
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        assert "." in parts[1], "time field must carry sub-second precision (R16)"
        low, high = float(parts[2]), float(parts[3])
        assert low >= 100.0e6 - 1e4 and high <= 101.0e6 + 1e4, "CSV trimmed to range (R11)"
    times, freqs, P = report.load(str(csv))
    assert freqs[0] >= 100.0e6 - 1e4 and freqs[-1] <= 101.0e6 + 1e4


def test_rate_coercion_uses_actual_rate_for_metadata(monkeypatch, capsys):
    # R06: when UHD coerces the rate, the emitted bin width must be actual/nfft.
    actual = 1.30e6
    out = _run_capture(monkeypatch, capsys, BASE + ["--stream"], f_tone=100.6e6, actual_rate=actual)
    obj = json.loads(out.strip().splitlines()[-1])
    _, nfft, bin_res, _, _, _ = sweep.build_hop_plan(100.0e6, 101.0e6, actual, 10e3, 0.2)
    assert obj["bin"] == pytest.approx(bin_res)
    assert obj["bin"] != pytest.approx(1e4)     # not the requested rate's bin
