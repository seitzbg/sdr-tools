"""gaincal: R04 (no clean gain), R03 (bounded recv), R02 (service cleanup)."""
import sys
import time
import types

import numpy as np
import pytest

from sdr_sweep import gaincal


# --- R04: every tested gain clips -> no usable recommendation ----------------

def test_all_clipping_returns_no_recommendation():
    slopes, knee, knee_reached, overload, rec = gaincal.analyse(
        [20, 25, 30], [-80, -75, -70], [0.1, 0.2, 0.3], 0.001, 0.7, 5)
    assert overload == 20          # the lowest tested gain already clips
    assert rec is None             # ...so there is no clean gain to recommend
    assert knee_reached is False


def test_clean_range_still_recommends_a_gain():
    # Happy path is unchanged: a knee below a high overload yields a real number.
    gains = [0, 10, 20, 30, 40]
    floors = [-100, -100, -90, -80, -70]     # tracks gain from ~20 dB up
    clips = [0.0, 0.0, 0.0, 0.0, 0.5]        # overload only at 40
    _, _, knee_reached, overload, rec = gaincal.analyse(gains, floors, clips, 1e-3, 0.7, 5)
    assert overload == 40
    assert rec is not None
    assert gains[0] <= rec < overload


# --- R03: the receive loop is bounded, not an infinite retry -----------------

class _MD:
    def strerror(self):
        return "timeout"


class _Streamer:
    """recv() returns the pre-scripted sample counts; 0 == a stalled burst."""
    def __init__(self, counts):
        self.counts = list(counts)

    def recv(self, recv_buf, md, timeout):
        n = self.counts.pop(0) if self.counts else 0
        if n > 0:
            recv_buf[0, :n] = 1.0
        return n


def test_burst_recv_returns_on_complete_burst():
    buf = np.zeros((1, 64), dtype=np.complex64)
    iq = gaincal._burst_recv(_Streamer([10]), buf, _MD(), 10, 0.1,
                             deadline=time.monotonic() + 10)
    assert iq.shape == (10,)
    assert np.allclose(iq, 1.0)


def test_burst_recv_raises_on_stall():
    buf = np.zeros((1, 64), dtype=np.complex64)
    with pytest.raises(RuntimeError, match="stalled"):
        gaincal._burst_recv(_Streamer([0]), buf, _MD(), 10, 0.01,
                            deadline=time.monotonic() + 10)


def test_burst_recv_raises_on_deadline():
    buf = np.zeros((1, 64), dtype=np.complex64)
    # recv keeps returning a trickle but never completes; a past deadline stops it.
    with pytest.raises(RuntimeError, match="deadline"):
        gaincal._burst_recv(_Streamer([1] * 100), buf, _MD(), 10, 0.01,
                            deadline=time.monotonic() - 1)


# --- R02: Ctrl-C during startup still restarts the monitoring service --------

class _Recorder:
    def __init__(self):
        self.calls = []

    def run(self, argv, **kw):
        self.calls.append(argv)
        return types.SimpleNamespace(returncode=0)


def test_interrupt_during_startup_restarts_service(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(gaincal, "subprocess", types.SimpleNamespace(run=rec.run))

    fake_time = types.SimpleNamespace(
        sleep=lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()),
        monotonic=time.monotonic)
    monkeypatch.setattr(gaincal, "time", fake_time)

    # A fake `uhd` so _require_uhd() (module import only) succeeds before the stop.
    monkeypatch.setitem(sys.modules, "uhd", types.ModuleType("uhd"))
    monkeypatch.setattr(sys, "argv", ["sdr-sweep-gaincal", "--manage-service"])

    with pytest.raises(KeyboardInterrupt):
        gaincal.main()

    stops = [c for c in rec.calls if c[:2] == ["systemctl", "stop"]]
    starts = [c for c in rec.calls if c[:2] == ["systemctl", "start"]]
    assert stops, "service should have been stopped"
    assert starts, "service MUST be restarted even when interrupted during startup"


# --- R09: bad options are rejected before anything is touched ----------------

def test_bad_gain_step_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sdr-sweep-gaincal", "--gain-step", "0"])
    with pytest.raises(SystemExit):
        gaincal.main()
