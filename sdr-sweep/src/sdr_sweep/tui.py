"""Live spectrum + waterfall TUI for the sdr-sweep JSON stream.

Reads one JSON object per sweep on stdin, so it runs anywhere uv is — pipe it
straight off the radio over SSH:

    ssh root@sdr1 'sdr/sdr_sweep.py --start 902M --stop 928M --stream --quiet' \\
        | ./sweep_tui.py

    # or replay a run you already streamed to a file
    ./sweep_tui.py < stream.jsonl

Shows a running spectrum (peak-hold), a scrolling colour waterfall, and live
stats (noise floor, strongest signals, active-bin count). Ctrl-C to quit.
"""
import json
import shutil
import sys
from collections import deque

import numpy as np

from sdr_sweep import __version__

BLOCKS = " ▁▂▃▄▅▆▇█"
# Coarse viridis anchors (t: 0..1) for the waterfall colour ramp.
_VIRIDIS = np.array([
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
    (181, 222, 43), (253, 231, 37)], dtype=float)


def color(t):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    x = t * (len(_VIRIDIS) - 1)
    i = int(x)
    if i >= len(_VIRIDIS) - 1:
        r, g, b = _VIRIDIS[-1]
    else:
        f = x - i
        r, g, b = _VIRIDIS[i] * (1 - f) + _VIRIDIS[i + 1] * f
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def pool(db, w):
    """Max-pool a full-res spectrum into w columns (max keeps narrow signals)."""
    if len(db) <= w:
        return db
    idx = np.linspace(0, len(db), w + 1).astype(int)
    return np.array([db[idx[i]:idx[i + 1]].max() for i in range(w)])


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Live spectrum + waterfall TUI for the sdr-sweep JSON stream")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.parse_args()
    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError as exc:
        raise SystemExit(
            "rich is required for the live TUI. Install it with `pip install 'sdr-sweep[tui]'`."
        ) from exc
    console = Console()
    peak_hold = None
    occ = None                       # EMA of active-bin fraction, per column
    waterfall = deque(maxlen=18)
    n = 0
    margin = 6.0

    def render(obj):
        nonlocal peak_hold, occ, n
        n += 1
        db = np.array(obj["db"], dtype=float)
        f0, f1, binhz = obj["f0"], obj["f1"], obj["bin"]
        floor = float(np.median(db))
        active = db > floor + margin
        peaki = int(np.argmax(db))
        peakf = f0 + peaki * binhz

        cols = max(40, min(shutil.get_terminal_size((120, 40)).columns - 4, 200))
        pdb = pool(db, cols)
        if peak_hold is None or len(peak_hold) != cols:
            peak_hold = pdb.copy()
            occ = pool(active.astype(float), cols)
        else:
            peak_hold = np.maximum(peak_hold * 0.995, pdb)   # slow-decay peak hold
            occ = 0.95 * occ + 0.05 * pool(active.astype(float), cols)

        vmin = float(np.percentile(db, 20))
        vmax = max(vmin + 6.0, float(np.percentile(db, 99.5)))

        # Spectrum row (bars, coloured by level) with the peak-hold ceiling.
        spec = Text()
        ph = Text()
        for i in range(cols):
            t = (pdb[i] - vmin) / (vmax - vmin)
            lvl = int(np.clip(t, 0, 1) * (len(BLOCKS) - 1))
            spec.append(BLOCKS[lvl], style=color(t))
            th = (peak_hold[i] - vmin) / (vmax - vmin)
            ph.append("·" if th > 0.15 else " ", style=color(th))

        # Waterfall: newest row on top.
        row = Text()
        for i in range(cols):
            t = (pdb[i] - vmin) / (vmax - vmin)
            row.append(" ", style=f"on {color(t)}")
        waterfall.appendleft(row)

        # Stats table.
        st = Table.grid(padding=(0, 2))
        st.add_column(justify="right", style="bold cyan")
        st.add_column()
        st.add_row("range", f"{f0/1e6:.3f}–{f1/1e6:.3f} MHz  ({binhz/1e3:.2f} kHz bins)")
        st.add_row("sweeps", str(n))
        st.add_row("floor", f"{floor:.1f} dB (uncal)")
        st.add_row("peak", f"{db[peaki]:.1f} dB @ {peakf/1e6:.4f} MHz (+{db[peaki]-floor:.1f})")
        st.add_row("active", f"{int(active.sum())}/{len(db)} bins > floor+{margin:.0f}dB")

        axis = Text(f"{f0/1e6:8.2f}{'':^{max(0,cols-18)}}{f1/1e6:8.2f}", style="dim")
        return Group(
            Panel(st, title="sdr sweep — live", border_style="cyan"),
            Text("spectrum (peak-hold ·):", style="dim"),
            ph, spec, axis,
            Text("waterfall (newest top):", style="dim"),
            *waterfall,
        )

    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                live.update(render(obj), refresh=True)
        except (KeyboardInterrupt, BrokenPipeError):
            pass


if __name__ == "__main__":
    main()
