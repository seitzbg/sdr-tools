#!/usr/bin/env python3
"""On-demand RX gain calibration for the USRP B210 (sdr1).

Sweeps RX gain across the band, measures the noise floor and the ADC clip
fraction at each step, then computes the optimal gain: high enough to be
external-noise-limited (full sensitivity — the floor tracks gain ~1:1) yet low
enough to stay clear of ADC overload on strong bursts. Prints a table + the
recommendation; changes nothing on its own.

OBSERVATION TIME IS THE WHOLE GAME for the clip half of this. The floor is
stationary and a millisecond of RF measures it fine; overload is bursty and a
millisecond measures nothing. Before --clip-dwell existed this tool looked at
~460 us of RF per gain, so on sdr1 it never saw a 1 W LoRa gateway keying up
for ~600 ms once a minute, reported clip=0, and recommended a gain with 17 dB
of headroom that did not exist. It now dwells on each hop with contiguous
captures and PRINTS how much RF it actually watched. A clip of 0 only ever
means "no overload in N seconds" — read the N.

For a definitive answer prefer the sweep exporter's continuous
sdr_sweep_clip_fraction_max, which sees every sweep forever; this tool is a
point-in-time probe.

Runs under sdr1's system python3 (uhd + numpy). It needs exclusive access to
the radio, so either stop the monitor first or pass --manage-service (stops
sdr-sweep-exporter for the measurement and restarts it afterwards, always).

    # simplest: let it borrow the radio from the running monitor
    /root/sdr/sweep_gaincal.py --manage-service

    # scriptable: just the recommended integer dB
    /root/sdr/sweep_gaincal.py --manage-service --json | jq .recommended
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np

from sdr_sweep import __version__, sweep

SERVICE = "sdr-sweep-exporter"


def _require_uhd():
    """Import UHD, or exit with a clear message (it is a system package, not on PyPI)."""
    try:
        import uhd
        return uhd
    except ImportError as exc:  # pragma: no cover - depends on system UHD
        raise SystemExit(
            "The 'uhd' Python module is required for radio capture but was not found. "
            "It ships with your platform's UHD install (e.g. `apt install uhd-host "
            "python3-uhd`), not from PyPI. See the README > Install. "
            "The analysis and exporter commands do not need it."
        ) from exc


def parse_hz(s):
    s = str(s).strip().lower()
    mult = 1.0
    if s and s[-1] in "kmg":
        mult = {"k": 1e3, "m": 1e6, "g": 1e9}[s[-1]]
        s = s[:-1]
    return float(s) * mult


def clip_fraction(iq):
    """fc32 RX is scaled to ~[-1, 1]; |I| or |Q| at ~1.0 = ADC saturation."""
    return float(np.mean((np.abs(iq.real) >= 0.99) | (np.abs(iq.imag) >= 0.99)))


def _burst_recv(streamer, recv_buf, md, nsamps, timeout, deadline):
    """Bounded burst receive into a fresh IQ array; UHD-type-free (testable).

    The caller has already issued the num_done stream command; this drains it
    with an explicit deadline and per-recv timeout so a wedged receiver cannot
    loop forever — the failure mode UHD's own recv_num_samps() convenience
    method has, since it retries until its sample count is reached and a stream
    of zero-sample timeouts never terminates it.
    """
    iq = np.empty(nsamps, dtype=np.complex64)
    got = 0
    while got < nsamps:
        n = streamer.recv(recv_buf, md, timeout)
        if not n:
            raise RuntimeError(f"rx burst stalled at {got}/{nsamps} samples: {md.strerror()}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"rx burst exceeded its deadline at {got}/{nsamps} samples")
        take = min(nsamps - got, n)
        iq[got:got + take] = recv_buf[0, :take]
        got += take
    return iq


def _burst(usrp, streamer, recv_buf, nsamps, samp_rate, timeout=0.5):
    """Issue a num_done burst for exactly `nsamps` and drain it with a deadline."""
    import uhd
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = int(nsamps)
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)
    md = uhd.types.RXMetadata()
    # Allow up to ~4x the burst's own duration plus a couple of recv timeouts
    # before declaring it wedged — generous for retune/settling, bounded overall.
    deadline = time.monotonic() + nsamps / samp_rate * 4.0 + 2.0 * timeout + 1.0
    return _burst_recv(streamer, recv_buf, md, int(nsamps), timeout, deadline)


def capture(usrp, streamer, recv_buf, center, gain, samp_rate, nsamps, timeout=0.5):
    """Tune + set gain (stream stopped), then a single bounded num_done burst."""
    import uhd
    usrp.set_rx_freq(uhd.types.TuneRequest(center), 0)
    usrp.set_rx_gain(gain, 0)
    return _burst(usrp, streamer, recv_buf, nsamps, samp_rate, timeout)


def probe_clip(usrp, streamer, recv_buf, center, samp_rate, gain, dwell_s, chunk_s):
    """Worst clip fraction over ~dwell_s of near-contiguous IQ at one center.

    Returns (worst_clip, rf_seconds_observed).

    THIS IS THE WHOLE POINT OF THE TOOL AND IT USED TO BE BROKEN. The old
    code took one (frames+1)*nfft capture per hop per gain — about 230 us of
    RF, ~460 us across two hops. Against a bursty overloader (a 1 W LoRa
    gateway keying up for ~600 ms roughly once a minute) the chance of
    landing on a burst is negligible, so it reported clip=0, concluded the
    overload onset was 20 dB higher than it is, and recommended a gain whose
    headroom did not exist. Measured on sdr1 2026-07-21: the sweep exporter's
    windowed metric saw ~1% of sweeps clipping at the gain gaincal called
    clean.

    Each chunk here is CONTIGUOUS, so a burst starting inside one is actually
    caught, and we keep going until the dwell elapses. Each burst is deadline-
    bounded (see _burst) so a stalled receiver aborts calibration loudly
    instead of hanging with the monitoring service left stopped.
    """
    n_chunk = max(1024, int(samp_rate * chunk_s))
    deadline = time.monotonic() + dwell_s
    worst, rf = 0.0, 0.0
    while True:
        iq = capture(usrp, streamer, recv_buf, center, gain, samp_rate, n_chunk)
        worst = max(worst, clip_fraction(iq))
        rf += len(iq) / samp_rate
        if time.monotonic() >= deadline:
            return worst, rf


def measure(usrp, streamer, recv_buf, centers, samp_rate, nfft, keep, frames, window, wnorm, gain,
            clip_dwell=0.0, clip_chunk_s=0.05):
    """Return (floor_db, clip_fraction, rf_seconds) for one gain.

    floor = mean across hops of the median PSD bin. The floor is stationary,
    so one short capture per hop is genuinely enough for it.

    clip = WORST across hops. Clipping is NOT stationary, so when clip_dwell
    is set we additionally dwell on each hop (see probe_clip). rf_seconds is
    how much RF we actually looked at — report it, because a clip reading is
    only ever a lower bound set by that number.
    """
    floors, clips = [], []
    rf_total = 0.0
    lo = (nfft - keep) // 2
    for c in centers:
        iq = capture(usrp, streamer, recv_buf, c, gain, samp_rate, (frames + 1) * nfft)
        iq = iq[nfft:][: frames * nfft]
        clips.append(clip_fraction(iq))
        rf_total += len(iq) / samp_rate
        mat = iq.reshape(frames, nfft)
        spec = np.fft.fftshift(np.fft.fft(mat * window, axis=1), axes=1)
        power = (np.abs(spec) ** 2).mean(axis=0) / wnorm
        db = 10.0 * np.log10(power + 1e-20)
        floors.append(float(np.median(db[lo:lo + keep])))
        if clip_dwell > 0:
            worst, rf = probe_clip(usrp, streamer, recv_buf, c, samp_rate, gain,
                                   clip_dwell, clip_chunk_s)
            clips.append(worst)
            rf_total += rf
    return float(np.mean(floors)), max(clips), rf_total


def analyse(gains, floors, clips, clip_thresh, knee_slope, step):
    slopes = [None] + [(floors[i] - floors[i - 1]) / (gains[i] - gains[i - 1])
                       for i in range(1, len(gains))]
    # Overload onset = lowest gain whose clip fraction exceeds the threshold.
    overload_idx = next((i for i, c in enumerate(clips) if c > clip_thresh), None)
    overload = gains[overload_idx] if overload_idx is not None else None
    # If the LOWEST tested gain already clips there is no clean operating point
    # at all: subtracting 10 dB and clamping back would "recommend" that same
    # overloaded gain. Return an explicit no-recommendation instead of publishing
    # a usable-looking number. The caller advises lower gain / more attenuation.
    if overload_idx == 0:
        return slopes, None, False, overload, None
    hi_idx = overload_idx if overload_idx is not None else len(gains)

    # Knee = lowest gain from which the floor tracks gain ~1:1 for the REST of
    # the clean range (external-noise-limited; below it the receiver's own
    # noise/quantisation dominates and the floor stays flat).
    #
    # This used to break on the FIRST step whose slope crossed the threshold,
    # which one noisy floor sample can trigger arbitrarily low. That is how a
    # knee of "~20 dB" came to be recorded for a receiver later measured at
    # slope 0.38 between 30 and 40 dB.
    #
    # Requiring EVERY step to clear the threshold is the same single-sample
    # brittleness pointing the other way: on sdr1's first real dwelling run the
    # floor tracked gain ~1:1 from 40 dB up (1.11 / 0.81 / 1.30) but a lone
    # 0.64 step at 50->55 failed an all() test, so the tool declared "no knee"
    # and jumped to the no-knee branch, recommending 50 dB. Measured, 40->50
    # has an overall slope of 1.05 -- 50 dB buys ZERO extra SNR over 40 while
    # costing 10 dB of clip headroom. One noisy step must not flip the whole
    # recommendation MODE like that.
    #
    # A plain mean over the remaining range is not the answer either: a long
    # 1:1 tail drags the average up and finds the knee far too low on a sharp
    # transition (a clean knee at 20 dB comes out as 5).
    #
    # So smooth the slopes with a 3-point moving median first -- that removes
    # an isolated dip or spike without blurring a genuine step -- then apply
    # the strict "holds for the rest of the clean range" test to the smoothed
    # series.
    sm = []
    for i in range(len(slopes)):
        w = [s for s in slopes[max(1, i - 1):i + 2] if s is not None]
        sm.append(sorted(w)[len(w) // 2] if w else None)
    knee, knee_reached = None, False
    for i in range(1, hi_idx):
        seg = [s for s in sm[i:hi_idx] if s is not None]
        if seg and all(s >= knee_slope for s in seg):
            knee, knee_reached = gains[i - 1], True
            break
    if not knee_reached:
        knee = gains[max(hi_idx - 1, 0)]

    hi = overload if overload is not None else gains[-1]
    lo = knee
    # Sit in the LOWER THIRD of the clean [knee, overload] range: anywhere above
    # the knee is fully external-noise-limited (identical detection SNR — gain
    # only shifts the floor, not the signal-to-floor ratio), so the tie-breaker
    # is headroom. Lower third keeps solid sensitivity while maximising the
    # margin before strong bursts clip the ADC (fewer false signals).
    if knee_reached:
        rec = lo + (hi - lo) / 3.0
        if hi - 10 > lo + 5:
            rec = min(max(rec, lo + 5), hi - 10)
    else:
        # Never external-noise-limited anywhere in the measured range: the
        # floor does NOT track gain, so every dB of gain given up costs real
        # detection SNR (sdr1 measured 3.8 dB of floor for a 10 dB gain cut —
        # slope 0.38 — i.e. ~7 dB of SNR lost going 40 -> 30). The "lower
        # third" rule assumes a knee exists and is actively harmful here, so
        # sit as HIGH as the clip headroom allows instead.
        rec = (hi - 10) if overload is not None else gains[-1]
    rec = int(round(rec / step) * step)
    rec = max(gains[0], min(gains[-1], rec))
    return slopes, knee, knee_reached, overload, rec


def main():
    ap = argparse.ArgumentParser(description="USRP B210 RX gain calibration")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--start", type=parse_hz, default=902e6)
    ap.add_argument("--stop", type=parse_hz, default=928e6)
    ap.add_argument("--hop-bw", type=parse_hz, default=20e6)
    ap.add_argument("--bin", type=parse_hz, default=40e3, help="FFT bin width Hz (default 40k — coarse is fine for cal)")
    ap.add_argument("--ant", default="RX2")
    ap.add_argument("--crop", type=float, default=0.2)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--gain-start", type=float, default=0.0)
    ap.add_argument("--gain-stop", type=float, default=None, help="default = device max")
    ap.add_argument("--gain-step", type=float, default=5.0)
    ap.add_argument("--clip-thresh", type=float, default=1e-3, help="clip fraction = overloaded (default 0.001)")
    ap.add_argument("--clip-dwell", type=float, default=2.0,
                    help="SECONDS of contiguous IQ to watch per hop per gain for clipping "
                         "(default 2.0). 0 = legacy single-shot, which sees ~230 us per hop "
                         "and CANNOT detect a bursty overloader — do not trust clip=0 from it.")
    ap.add_argument("--clip-chunk-ms", type=float, default=50.0,
                    help="contiguous capture length per read during the dwell (default 50 ms)")
    ap.add_argument("--knee-slope", type=float, default=0.7, help="floor-vs-gain slope for external-noise-limited")
    ap.add_argument("--manage-service", action="store_true",
                    help="stop %s for the measurement and restart it after (always)" % SERVICE)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    a = ap.parse_args()

    def note(*m):
        if not a.json:
            print(*m, file=sys.stderr, flush=True)

    def warn(*m):
        # Always visible, even with --json: a failed stop/restart means the
        # monitoring service state is not what the caller assumes.
        print(*m, file=sys.stderr, flush=True)

    # Validate options and dependency availability BEFORE taking monitoring
    # offline, so a bad argument or a missing UHD module never leaves the
    # exporter stopped.
    if a.gain_step <= 0:
        ap.error("--gain-step must be > 0")
    if a.gain_stop is not None and a.gain_stop <= a.gain_start:
        ap.error("--gain-stop must be greater than --gain-start")
    try:
        sweep.validate_scan(a.hop_bw, a.bin, a.crop, a.frames)
    except ValueError as exc:
        ap.error(str(exc))
    uhd = _require_uhd()  # module import only; the device is opened after the stop

    managed = False
    try:
        # Enter the try (so `finally` will always restart the service) BEFORE
        # stopping it: a Ctrl-C during the stop or the settle sleep must still
        # restore monitoring.
        if a.manage_service:
            managed = True
            stop = subprocess.run(["systemctl", "stop", SERVICE])
            if stop.returncode != 0:
                warn(f"[gaincal] WARNING: 'systemctl stop {SERVICE}' returned "
                     f"{stop.returncode}; it may still be holding the radio")
            else:
                note(f"[gaincal] paused {SERVICE}")
            time.sleep(2)

        samp_rate, nfft, bin_hz, keep, centers, usable = sweep.build_hop_plan(
            a.start, a.stop, a.hop_bw, a.bin, a.crop)

        usrp = uhd.usrp.MultiUSRP()
        usrp.set_rx_antenna(a.ant, 0)
        usrp.set_rx_rate(samp_rate, 0)
        # UHD may coerce the rate (get_rx_rate is the source of truth). Rebuild
        # the plan on the actual rate so bin width, hop spacing and the reported
        # RF seconds describe the spectrum actually sampled.
        actual_rate = float(usrp.get_rx_rate(0))
        if abs(actual_rate - samp_rate) > 1.0:
            note(f"[gaincal] UHD coerced sample rate {samp_rate/1e6:.6f}M -> "
                 f"{actual_rate/1e6:.6f}M; rebuilding the plan on the actual rate")
            samp_rate, nfft, bin_hz, keep, centers, usable = sweep.build_hop_plan(
                a.start, a.stop, actual_rate, a.bin, a.crop)
        window = np.hanning(nfft)
        wnorm = (window ** 2).sum()

        grange = usrp.get_rx_gain_range(0)
        gmax = a.gain_stop if a.gain_stop is not None else grange.stop()
        gmin = max(a.gain_start, grange.start())
        st = uhd.usrp.StreamArgs("fc32", "sc16")
        st.channels = [0]
        streamer = usrp.get_rx_stream(st)
        recv_buf = np.zeros((1, streamer.get_max_num_samps()), dtype=np.complex64)

        gains = list(np.arange(gmin, gmax + 0.1, a.gain_step))
        note(f"[gaincal] {a.start/1e6:.0f}-{a.stop/1e6:.0f} MHz  ant={a.ant}  "
             f"gains {gmin:.0f}..{gmax:.0f}/{a.gain_step:.0f} dB  hops={len(centers)}")
        if a.clip_dwell <= 0:
            note("[gaincal] WARNING --clip-dwell 0: clip detection is single-shot "
                 "(~230 us/hop). A clip of 0 means 'not seen', NOT 'not clipping'.")
        floors, clips, rfsecs = [], [], []
        for g in gains:
            f, c, rf = measure(usrp, streamer, recv_buf, centers, samp_rate, nfft, keep,
                               a.frames, window, wnorm, g, a.clip_dwell, a.clip_chunk_ms / 1000.0)
            floors.append(f)
            clips.append(c)
            rfsecs.append(rf)
            note(f"  gain {g:5.1f} dB  floor {f:7.2f} dB  clip {c*100:6.3f}%  "
                 f"(watched {rf:.2f}s RF)")

        rf_total = sum(rfsecs)
        slopes, knee, knee_reached, overload, rec = analyse(
            gains, floors, clips, a.clip_thresh, a.knee_slope, a.gain_step)
    finally:
        if managed:
            start = subprocess.run(["systemctl", "start", SERVICE])
            if start.returncode != 0:
                warn(f"[gaincal] WARNING: failed to restart {SERVICE} (rc={start.returncode}) — "
                     f"monitoring may be DOWN; run 'systemctl start {SERVICE}'")
            else:
                note(f"[gaincal] restarted {SERVICE}")

    if a.json:
        print(json.dumps({
            "recommended": rec, "knee": knee, "knee_reached": knee_reached,
            # False when every tested gain clips: `recommended` is null and the
            # caller should lower gain or add attenuation. See analyse().
            "clean_gain_exists": rec is not None,
            "overload_onset": overload,
            "gain_min": gmin, "gain_max": gmax,
            # How much RF was actually inspected. A clip_frac of 0 is only ever
            # "no overload seen in this many seconds" — publish the seconds so
            # nobody reads it as "no overload exists".
            "clip_observed_seconds_total": round(rf_total, 3),
            "clip_dwell_s": a.clip_dwell,
            "points": [{"gain": g, "floor_db": round(f, 2), "clip_frac": round(c, 6),
                        "clip_observed_seconds": round(rf, 3),
                        "slope": (round(s, 3) if s is not None else None)}
                       for g, f, c, s, rf in zip(gains, floors, clips, slopes, rfsecs)]}))
        return

    print("\n  gain    floor      slope    clip%")
    for g, f, s, c in zip(gains, floors, slopes, clips):
        ss = f"{s:5.2f}" if s is not None else "   — "
        flag = "  <-- clipping" if c > a.clip_thresh else ""
        print(f"  {g:5.1f}  {f:8.2f}   {ss}   {c*100:7.3f}{flag}")
    if knee_reached:
        print(f"\n  sensitivity knee : ~{knee:.0f} dB (external-noise-limited at/above this)")
    else:
        print(f"\n  sensitivity knee : NOT REACHED in {gmin:.0f}..{gmax:.0f} dB — the floor "
              f"never tracks gain 1:1 here, so LOWERING gain costs real detection SNR")
    if overload is not None:
        print(f"  overload onset   : ~{overload:.0f} dB")
    else:
        print(f"  overload onset   : none seen in {rf_total:.1f}s of RF "
              f"(NOT proof of none — see the caveat below)")
    if rec is None:
        print(f"  >> recommended    : NONE — the lowest tested gain ({gmin:.0f} dB) already "
              f"clips.\n     No clean operating point exists in this range. Lower --gain-start, "
              f"add\n     external attenuation, or reduce the input signal, then re-run.")
    elif knee_reached:
        print(f"  >> recommended    : {rec} dB  (lower third of the clean range: "
              f"full sensitivity + maximum clip headroom)")
    else:
        print(f"  >> recommended    : {rec} dB  (as high as clip headroom allows — below the "
              f"knee every dB you give up costs SNR)")

    print("\n  The knee is a heuristic fitted to ONE noisy pass — the floor scatter here")
    print("  is a dB or two, which is the same size as the slope differences it keys on.")
    print("  Treat the recommendation as advisory. The authoritative test is an A/B: set")
    print("  two candidate gains and compare peak-minus-floor from the exporter's windowed")
    print("  series over minutes, which averages far more data than this pass can.")

    print(f"\n  clip observed over {rf_total:.1f}s of RF total, {a.clip_dwell:.1f}s dwell/hop/gain.")
    if a.clip_dwell <= 0:
        print("  !! --clip-dwell 0 sees ~230us per hop. A bursty overloader (e.g. a LoRa")
        print("     gateway keying up ~600ms once a minute) will be MISSED and this tool")
        print("     will recommend headroom that does not exist. Use the default dwell.")
    else:
        per_gain = rf_total / max(len(gains), 1)
        print(f"  Each gain watched ~{per_gain:.1f}s, so a burst recurring every T seconds is")
        print(f"  caught at that gain with probability roughly {per_gain:.1f}/T — e.g. ~"
              f"{100.0 * min(per_gain / 60.0, 1.0):.0f}% for a once-a-minute burst.")
        print("  Raise --clip-dwell for a rarer emitter, or just trust the exporter's")
        print("  continuous sdr_sweep_clip_fraction_max — it watches every sweep, forever.")
    if rec is not None:
        print(f"\n  To apply: set  sweep_gain: \"{rec}\"  in munro/ansible "
              f"playbooks/sdr1-sweep-exporter.yml and re-run it.")


if __name__ == "__main__":
    main()
