#!/usr/bin/env python3
"""Prometheus exporter for sdr_sweep.py's JSON stream — stdlib only.

Reads one JSON object per sweep on stdin and serves Prometheus text metrics on
:PORT/metrics, so the fleet Prometheus can scrape long-term noise-floor and
occupancy trends for a parked band. Runs under sdr1's system python3 (no uv, no
third-party deps).

    # park on the ISM band and export forever
    /root/sdr/sdr_sweep.py --start 902M --stop 928M --stream --quiet \\
        | /root/sdr/sweep_exporter.py --port 9821 --bands 8

Then add a scrape job pointing at sdr1:9821 (see README — that wiring is left
to you since it lives in the ansible-managed prometheus config).

Sweeps arrive far faster than Prometheus scrapes: the parked 902-928 MHz config
produces ~279 sweeps/min, so a 30s scrape sees ~139 sweeps and the plain
`sdr_sweep_*` gauges (latest sweep) publish exactly one of them. That is fine
for slow-moving quantities like the noise floor, but it makes the interesting
ones — peak level, occupancy, clip fraction — a 1-in-139 coin flip: a burst or
an ADC overload is almost certainly discarded.

So alongside the latest-sweep gauges we keep a rolling `--window` (default 60s,
~2x the scrape interval) of per-sweep summaries and publish `_max`/`_mean`
companions over it. Those see every sweep. The window is time-based and pruned
on both ingest and render, so rendering is idempotent — scraping twice, or
curling /metrics by hand while Prometheus is also scraping, changes nothing.
"""
import argparse
import collections
import json
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sdr_sweep import __version__

_LOCK = threading.Lock()
_STATE = {"have": False}
_MARGIN = 6.0
_NBANDS = 8
_WINDOW = 60.0
# (monotonic_t, summary) per sweep, pruned to _WINDOW. Only scalars + per-band
# summaries are kept — never the raw `db` array, which would be ~1330 floats x
# ~279 sweeps in the window.
_SWEEPS = collections.deque(maxlen=100_000)


def update(obj):
    db = obj["db"]
    if not db:
        return
    f0, f1, binhz = obj["f0"], obj["f1"], obj["bin"]
    floor = statistics.median(db)
    thr = floor + _MARGIN
    active = sum(1 for v in db if v > thr)
    peak = max(db)
    peak_i = db.index(peak)
    peak_f = f0 + peak_i * binhz

    bands = []
    if _NBANDS > 0:
        n = len(db)
        for b in range(_NBANDS):
            lo, hi = b * n // _NBANDS, (b + 1) * n // _NBANDS
            seg = db[lo:hi]
            if not seg:
                continue
            blo = f0 + lo * binhz
            bhi = f0 + hi * binhz
            bthr = statistics.median(seg) + _MARGIN
            bands.append({
                "label": f"{blo/1e6:.1f}-{bhi/1e6:.1f}MHz",
                "peak": max(seg),
                "occ": sum(1 for v in seg if v > bthr) / len(seg),
            })

    summary = {
        "floor": floor, "peak": peak, "peak_f": peak_f,
        "active": active, "total": len(db), "occ": active / len(db),
        "clip": float(obj.get("clip", 0.0)), "bands": bands,
    }

    with _LOCK:
        _STATE.update({
            "have": True, "f0": f0, "f1": f1, "t": obj.get("t", 0.0),
            "sweeps": _STATE.get("sweeps", 0) + 1, **summary,
        })
        # Window bookkeeping uses the monotonic clock, not the sweep's own `t`:
        # `t` is producer wall-clock and an NTP step would otherwise flush or
        # freeze the window.
        _SWEEPS.append((time.monotonic(), summary))
        _prune()


def _prune():
    """Drop sweeps older than _WINDOW. Caller must hold _LOCK."""
    cutoff = time.monotonic() - _WINDOW
    while _SWEEPS and _SWEEPS[0][0] < cutoff:
        _SWEEPS.popleft()


def render_metrics():
    with _LOCK:
        s = dict(_STATE)
        # Prune here too: if the producer stalls, the window must decay to
        # empty rather than serve a frozen aggregate forever.
        _prune()
        win = [sm for _, sm in _SWEEPS]
    if not s.get("have"):
        return "# no sweep data received yet\n"
    out = []

    def g(name, help_, val, labels=""):
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} gauge")
        out.append(f"{name}{labels} {val}")

    g("sdr_sweep_noise_floor_db", "Median noise floor of latest sweep (uncal dB)", f"{s['floor']:.3f}")
    g("sdr_sweep_peak_db", "Strongest bin of latest sweep (uncal dB)", f"{s['peak']:.3f}")
    g("sdr_sweep_peak_freq_hz", "Frequency of the strongest bin (Hz)", f"{s['peak_f']:.0f}")
    g("sdr_sweep_active_bins", f"Bins above floor+{_MARGIN:.0f}dB in latest sweep", s["active"])
    g("sdr_sweep_total_bins", "Total bins in latest sweep", s["total"])
    g("sdr_sweep_occupancy_ratio", "active/total bins in latest sweep", f"{s['occ']:.4f}")
    g("sdr_sweep_clip_fraction", "IQ samples at ADC full-scale in latest sweep (overload, 0..1)",
      f"{s.get('clip', 0.0):.6f}")
    g("sdr_sweep_last_timestamp_seconds", "Unix time of latest sweep", f"{s['t']:.3f}")
    out.append("# HELP sdr_sweep_sweeps_total Sweeps processed since start")
    out.append("# TYPE sdr_sweep_sweeps_total counter")
    out.append(f"sdr_sweep_sweeps_total {s['sweeps']}")

    if s.get("bands"):
        out.append("# HELP sdr_sweep_band_peak_db Peak level per sub-band (uncal dB)")
        out.append("# TYPE sdr_sweep_band_peak_db gauge")
        for b in s["bands"]:
            out.append(f'sdr_sweep_band_peak_db{{band="{b["label"]}"}} {b["peak"]:.3f}')
        out.append("# HELP sdr_sweep_band_occupancy_ratio Occupancy per sub-band")
        out.append("# TYPE sdr_sweep_band_occupancy_ratio gauge")
        for b in s["bands"]:
            out.append(f'sdr_sweep_band_occupancy_ratio{{band="{b["label"]}"}} {b["occ"]:.4f}')

    # --- rolling-window aggregates -------------------------------------------
    # Every sweep in the window contributes, so short bursts and ADC overloads
    # survive to Prometheus instead of needing to land on the one sweep that
    # happened to be latest at scrape time.
    g("sdr_sweep_window_seconds", "Width of the aggregation window (s)", f"{_WINDOW:.1f}")
    g("sdr_sweep_window_sweeps", "Sweeps contributing to the *_max/*_mean series", len(win))
    if win:
        floors = [w["floor"] for w in win]
        occs = [w["occ"] for w in win]
        clips = [w["clip"] for w in win]
        top = max(win, key=lambda w: w["peak"])

        g("sdr_sweep_noise_floor_db_mean", f"Mean median-noise-floor over {_WINDOW:.0f}s (uncal dB)",
          f"{statistics.fmean(floors):.3f}")
        g("sdr_sweep_noise_floor_db_max", f"Worst (highest) noise floor over {_WINDOW:.0f}s (uncal dB)",
          f"{max(floors):.3f}")
        g("sdr_sweep_peak_db_max", f"Strongest bin seen over {_WINDOW:.0f}s (uncal dB)",
          f"{top['peak']:.3f}")
        g("sdr_sweep_peak_freq_hz_at_max", f"Frequency of the strongest bin over {_WINDOW:.0f}s (Hz)",
          f"{top['peak_f']:.0f}")
        g("sdr_sweep_active_bins_max", f"Most active bins in any sweep over {_WINDOW:.0f}s",
          max(w["active"] for w in win))
        g("sdr_sweep_occupancy_ratio_max", f"Highest occupancy of any sweep over {_WINDOW:.0f}s",
          f"{max(occs):.4f}")
        g("sdr_sweep_occupancy_ratio_mean", f"Mean occupancy over {_WINDOW:.0f}s",
          f"{statistics.fmean(occs):.4f}")
        g("sdr_sweep_clip_fraction_max", f"Worst ADC clip fraction over {_WINDOW:.0f}s (0..1)",
          f"{max(clips):.6f}")
        g("sdr_sweep_clip_fraction_mean", f"Mean ADC clip fraction over {_WINDOW:.0f}s (0..1)",
          f"{statistics.fmean(clips):.6f}")

        # Per-band, grouped by label. Bands are generated in ascending frequency
        # order every sweep, so first-seen insertion order is frequency order.
        agg = {}
        for w in win:
            for b in w["bands"]:
                slot = agg.setdefault(b["label"], {"peak": [], "occ": []})
                slot["peak"].append(b["peak"])
                slot["occ"].append(b["occ"])
        if agg:
            out.append(f"# HELP sdr_sweep_band_peak_db_max Peak level per sub-band over {_WINDOW:.0f}s (uncal dB)")
            out.append("# TYPE sdr_sweep_band_peak_db_max gauge")
            for label, v in agg.items():
                out.append(f'sdr_sweep_band_peak_db_max{{band="{label}"}} {max(v["peak"]):.3f}')
            out.append(f"# HELP sdr_sweep_band_occupancy_ratio_max Highest per-sub-band occupancy over {_WINDOW:.0f}s")
            out.append("# TYPE sdr_sweep_band_occupancy_ratio_max gauge")
            for label, v in agg.items():
                out.append(f'sdr_sweep_band_occupancy_ratio_max{{band="{label}"}} {max(v["occ"]):.4f}')
            out.append(f"# HELP sdr_sweep_band_occupancy_ratio_mean Mean per-sub-band occupancy over {_WINDOW:.0f}s")
            out.append("# TYPE sdr_sweep_band_occupancy_ratio_mean gauge")
            for label, v in agg.items():
                out.append(f'sdr_sweep_band_occupancy_ratio_mean{{band="{label}"}} {statistics.fmean(v["occ"]):.4f}')
    return "\n".join(out) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/metrics", "/"):
            body = render_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_):
        pass  # quiet


def read_stdin():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            update(json.loads(line))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue


def main():
    global _MARGIN, _NBANDS, _WINDOW
    ap = argparse.ArgumentParser(description="Prometheus exporter for sdr_sweep stream")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--port", type=int, default=9821)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--margin", type=float, default=6.0, help="dB over floor = active (default 6)")
    ap.add_argument("--bands", type=int, default=8, help="number of per-band metric buckets (0 = off)")
    ap.add_argument("--window", type=float, default=60.0,
                    help="rolling window in seconds for the *_max/*_mean series; set to at "
                         "least 2x the Prometheus scrape interval (default 60)")
    ap.add_argument("--exit-on-eof", action="store_true",
                    help="exit when the input stream closes (for the systemd service, so a "
                         "dead capture restarts the whole pipeline instead of serving stale data)")
    args = ap.parse_args()
    _MARGIN = args.margin
    _NBANDS = args.bands
    if args.window <= 0:
        ap.error("--window must be > 0")
    _WINDOW = args.window

    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"[exporter] serving http://{args.bind}:{args.port}/metrics", file=sys.stderr, flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        read_stdin()            # blocks until the stream closes
    except KeyboardInterrupt:
        return
    if args.exit_on_eof:
        print("[exporter] stdin EOF — exiting so the service pipeline restarts", file=sys.stderr)
        srv.shutdown()
        return
    print("[exporter] stdin EOF — still serving last metrics (Ctrl-C to stop)", file=sys.stderr)
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
