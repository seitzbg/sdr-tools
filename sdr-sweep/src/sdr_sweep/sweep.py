#!/usr/bin/env python3
"""Wideband power sweep for the USRP B210 (sdr1) — the capture engine.

Retunes across a frequency range in hops, FFTs each hop, crops the filter
roll-off edges, stitches the kept bins into one spectrum, and repeats over
time. Writes an rtl_power-compatible CSV log (one line per hop) and can also
emit one JSON object per completed sweep to stdout (--stream) for the live TUI
and Prometheus exporter to consume.

Runs under sdr1's *system* python3 — it needs the system `uhd` module and
numpy (both already present). No third-party deps, no uv. RX only (never TX).

Example — watch the 902-928 MHz ISM band for 10 minutes, ~10 kHz bins:
    ./sdr_sweep.py --start 902M --stop 928M --bin 10k --duration 600 \\
        -o /root/sweeps/ism_$(date +%F_%H%M).csv

Wide survey with a live TUI on your laptop:
    ssh root@sdr1 'sdr/sdr_sweep.py --start 400M --stop 2.5G --stream --quiet' \\
        | ./sweep_tui.py
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

from sdr_sweep import __version__

_STOP = False


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


def _on_sigint(_sig, _frm):
    global _STOP
    _STOP = True


def parse_hz(s):
    """Accept 902e6, 902M, 902.5M, 26M, 10k, 902000000."""
    s = str(s).strip().lower()
    mult = 1.0
    if s and s[-1] in "kmg":
        mult = {"k": 1e3, "m": 1e6, "g": 1e9}[s[-1]]
        s = s[:-1]
    return float(s) * mult


def next_pow2(x):
    return 1 << (int(math.ceil(x)) - 1).bit_length()


def build_plan(args):
    """Turn the requested range into a concrete hop plan.

    Returns (samp_rate, nfft, bin_hz, keep, centers, usable_bw). If the whole
    range fits in one hop bandwidth we park on a single center (fast cadence);
    otherwise we tile the range with contiguous cropped hops.
    """
    span = args.stop - args.start
    samp_rate = float(args.hop_bw)
    nfft = int(next_pow2(samp_rate / args.bin))
    bin_hz = samp_rate / nfft
    # Keep the central (1-crop) fraction of bins; the rest is analog/decimation
    # filter roll-off. Make it even so it's symmetric about DC.
    keep = int(nfft * (1.0 - args.crop)) & ~1
    usable_bw = keep * bin_hz

    if span <= usable_bw:
        centers = [args.start + span / 2.0]
    else:
        n_hops = int(math.ceil(span / usable_bw))
        centers = [args.start + usable_bw * (i + 0.5) for i in range(n_hops)]
    return samp_rate, nfft, bin_hz, keep, centers, usable_bw


def hop_psd(usrp, streamer, recv_buf, center, nfft, frames, window, wnorm):
    """Capture one hop; return (cropped DC-nulled power spectrum dB, clip_fraction).

    clip_fraction = fraction of I/Q samples at ~ADC full-scale (fc32 is scaled to
    [-1, 1]) — a direct, signal-agnostic overload indicator.

    Tunes with the stream stopped, then issues a `num_done` burst for exactly
    the samples needed — the device stops itself, so there is no stop_cont +
    drain cycle. The old recv_num_samps() path paid ~100 ms/hop there (its
    drain loop's final recv() always blocks for the full 0.1 s default
    timeout), which capped the sweep at ~4.7/s and a 0.8% RF duty cycle.
    Measured on the B210: 105.95 -> 4.18 ms/hop, spectra unchanged, zero
    stale pre-retune samples across a 15 dB band step, zero overflows over
    400 bursts. (A persistent start_cont stream is faster still but captures
    up to ~70k pre-retune samples after set_rx_freq returns, and overflows —
    do not "optimize" this back to one.)
    """
    import uhd
    # Grab one extra FFT frame and drop it — covers retune/AGC settling.
    nsamps = (frames + 1) * nfft
    usrp.set_rx_freq(uhd.types.TuneRequest(center), 0)
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = nsamps
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)
    iq = np.empty(nsamps, dtype=np.complex64)
    md = uhd.types.RXMetadata()
    got = 0
    while got < nsamps:
        n = streamer.recv(recv_buf, md, 0.5)
        if not n:
            # A wedged burst never completes on its own; die loudly so the
            # pipe EOF trips the exporter's --exit-on-eof and Restart=always
            # brings the pair back (a silent stall is invisible until the
            # window_sweeps==0 signal catches it much later).
            raise RuntimeError(
                f"rx burst stalled at {got}/{nsamps} samples: {md.strerror()}")
        take = min(nsamps - got, n)
        iq[got:got + take] = recv_buf[0, :take]
        got += take
    iq = iq[nfft:]  # drop settling frame
    iq = iq[: frames * nfft].reshape(frames, nfft)
    clip = float(np.mean((np.abs(iq.real) >= 0.99) | (np.abs(iq.imag) >= 0.99)))
    spec = np.fft.fftshift(np.fft.fft(iq * window, axis=1), axes=1)
    power = (np.abs(spec) ** 2).mean(axis=0) / wnorm
    db = 10.0 * np.log10(power + 1e-20)
    # Null the DC spike (center bin) with its neighbours' mean.
    c = nfft // 2
    db[c] = 0.5 * (db[c - 1] + db[c + 1])
    return db, clip


def main():
    ap = argparse.ArgumentParser(description="USRP B210 wideband power sweep")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--start", type=parse_hz, required=True, help="range start (e.g. 902M)")
    ap.add_argument("--stop", type=parse_hz, required=True, help="range stop (e.g. 928M)")
    ap.add_argument("--bin", type=parse_hz, default=10e3, help="target bin width Hz (default 10k)")
    ap.add_argument("--hop-bw", type=parse_hz, default=20e6, help="per-hop sample rate/bandwidth Hz (default 20M)")
    ap.add_argument("--gain", type=float, default=40.0, help="RX gain dB (default 40)")
    ap.add_argument("--ant", default="RX2", help="antenna (default RX2 = the RX1 SMA connector)")
    ap.add_argument("--crop", type=float, default=0.2, help="fraction of each hop's edges to discard (default 0.2)")
    ap.add_argument("--frames", type=int, default=16, help="FFT frames averaged per hop (default 16)")
    ap.add_argument("--stable-floor", action="store_true",
                    help="pin the AD9361 auto DC-offset/IQ-balance corrections OFF for a "
                         "steadier noise floor over long runs (adds a static DC spike — "
                         "already nulled — and faint mirror images of strong signals)")
    ap.add_argument("--interval", type=float, default=0.0, help="min seconds between sweep starts (default 0)")
    ap.add_argument("--duration", type=float, default=0.0, help="total run seconds (0 = until Ctrl-C)")
    ap.add_argument("--count", type=int, default=0, help="number of sweeps (0 = unlimited)")
    ap.add_argument("-o", "--output", default=None, help="rtl_power-format CSV log path")
    ap.add_argument("--stream", action="store_true", help="emit one JSON object per sweep to stdout")
    ap.add_argument("--quiet", action="store_true", help="suppress stderr progress")
    args = ap.parse_args()

    if args.stop <= args.start:
        ap.error("--stop must be greater than --start")

    samp_rate, nfft, bin_hz, keep, centers, usable_bw = build_plan(args)
    window = np.hanning(nfft)
    wnorm = (window ** 2).sum()

    def note(*a):
        if not args.quiet:
            print(*a, file=sys.stderr, flush=True)

    note(f"[sweep] {args.start/1e6:.3f}-{args.stop/1e6:.3f} MHz  "
         f"hop_bw={samp_rate/1e6:.1f}M  nfft={nfft}  bin={bin_hz/1e3:.3f}kHz  "
         f"keep={keep}/{nfft} bins  hops={len(centers)}  gain={args.gain}dB  ant={args.ant}")

    uhd = _require_uhd()
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_rx_antenna(args.ant, 0)
    usrp.set_rx_rate(samp_rate, 0)
    usrp.set_rx_gain(args.gain, 0)  # fixed for the run; hop_psd only retunes
    if args.stable_floor:
        # Stop the periodic re-convergence that wobbles the floor by ~1 dB.
        usrp.set_rx_dc_offset(False, 0)
        usrp.set_rx_iq_balance(False, 0)
        note("[sweep] stable-floor: AD9361 auto DC-offset/IQ-balance correction OFF")
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    streamer = usrp.get_rx_stream(st_args)
    recv_buf = np.zeros((1, streamer.get_max_num_samps()), dtype=np.complex64)

    logf = open(args.output, "a", buffering=1) if args.output else None
    signal.signal(signal.SIGINT, _on_sigint)

    t_run0 = time.time()
    n_done = 0
    try:
        while not _STOP:
            t0 = time.time()
            hop_db = []
            hop_clip = []
            for center in centers:
                if _STOP:
                    break
                db, clip = hop_psd(usrp, streamer, recv_buf, center,
                                   nfft, args.frames, window, wnorm)
                lo = (nfft - keep) // 2
                hop_db.append((center, db[lo:lo + keep]))
                hop_clip.append(clip)
            if _STOP:
                break

            # rtl_power CSV: one line per hop.
            if logf:
                d, t = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime(t0)).split(", ")
                for center, dbk in hop_db:
                    low = center - usable_bw / 2.0
                    high = center + usable_bw / 2.0
                    row = [d, t, f"{low:.0f}", f"{high:.0f}", f"{bin_hz:.2f}",
                           str(args.frames * nfft)]
                    row += [f"{v:.2f}" for v in dbk]
                    logf.write(", ".join(row) + "\n")

            # JSONL stream: whole stitched sweep, trimmed to [start, stop].
            if args.stream:
                freqs = np.concatenate([
                    center + (np.arange(keep) - keep / 2.0 + 0.5) * bin_hz
                    for center, _ in hop_db])
                allbins = np.concatenate([dbk for _, dbk in hop_db])
                m = (freqs >= args.start) & (freqs <= args.stop)
                obj = {"t": round(t0, 3), "f0": float(freqs[m][0]),
                       "f1": float(freqs[m][-1]), "bin": bin_hz,
                       "clip": round(max(hop_clip), 6) if hop_clip else 0.0,
                       "db": [round(float(v), 2) for v in allbins[m]]}
                try:
                    print(json.dumps(obj), flush=True)
                except BrokenPipeError:
                    break  # downstream (TUI/head) closed — stop cleanly

            n_done += 1
            note(f"[sweep] #{n_done} done in {time.time()-t0:.2f}s "
                 f"({len(centers)} hops)")

            if args.count and n_done >= args.count:
                break
            if args.duration and (time.time() - t_run0) >= args.duration:
                break
            if args.interval:
                slack = args.interval - (time.time() - t0)
                while slack > 0 and not _STOP:
                    time.sleep(min(slack, 0.2))
                    slack = args.interval - (time.time() - t0)
    finally:
        if logf:
            logf.close()
        note(f"[sweep] stopped after {n_done} sweeps, {time.time()-t_run0:.1f}s")
        # Avoid a noisy "Exception ignored while flushing sys.stdout" if the
        # stream consumer went away.
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


if __name__ == "__main__":
    main()
