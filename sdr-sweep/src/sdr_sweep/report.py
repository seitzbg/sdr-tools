"""Analyse an sdr-sweep CSV log: noise floor & drift, band occupancy, and
intermittent/persistent emitters. Emits a Markdown report and a PNG waterfall.

This is the offline half — it touches no hardware, so it runs anywhere uv is
installed (your laptop or sdr1). Power is uncalibrated dB (relative), which is
fine for noise-floor drift, occupancy and burst detection — all of which are
about *changes* and *contrast*, not absolute dBm.

    ./sweep_report.py /root/sweeps/ism.csv                 # report -> stdout, PNG next to csv
    ./sweep_report.py ism.csv --md report.md --png wf.png --margin 8
"""
import argparse
import sys
from datetime import datetime

import numpy as np

from sdr_sweep import __version__


def load(path):
    """Parse an rtl_power-format CSV into (times, freqs, P[time, freq]) in dB.

    Sweeps are delimited by the hop frequency wrapping back down to the start;
    that's robust even when several hops share the same wall-clock second.
    """
    lows, highs, step = [], [], None
    rows = []  # (epoch, low, np.array(db))
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            d, t = parts[0], parts[1]
            low, high, st = float(parts[2]), float(parts[3]), float(parts[4])
            db = np.array([float(x) for x in parts[6:]], dtype=np.float64)
            epoch = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S").timestamp()
            rows.append((epoch, low, high, st, db))
            lows.append(low)
            highs.append(high)
            step = st
    if not rows:
        sys.exit(f"no data rows in {path}")

    gmin, gmax = min(lows), max(highs)
    nbins = int(round((gmax - gmin) / step))
    freqs = gmin + (np.arange(nbins) + 0.5) * step

    # Group hops into sweeps: a new sweep starts when `low` stops increasing.
    sweeps, cur, prev_low = [], [], None
    for r in rows:
        if prev_low is not None and r[1] <= prev_low:
            sweeps.append(cur)
            cur = []
        cur.append(r)
        prev_low = r[1]
    if cur:
        sweeps.append(cur)

    times = np.empty(len(sweeps))
    P = np.full((len(sweeps), nbins), np.nan, dtype=np.float64)
    for si, sw in enumerate(sweeps):
        times[si] = sw[0][0]
        for epoch, low, high, st, db in sw:
            i0 = int(round((low - gmin) / step))
            P[si, i0:i0 + len(db)] = db
    return times, freqs, P


def analyse(times, freqs, P, margin, floor_pct):
    """Return a dict of derived metrics used by both the report and the plot."""
    # Per-bin baseline (quiet level) = low percentile across time. Robust to
    # signals that are present only part of the time.
    baseline = np.nanpercentile(P, floor_pct, axis=0)
    thresh = baseline + margin                      # per-bin "active" threshold
    active = P > thresh                             # bool [time, freq]

    # Overall noise floor per sweep = median across freq (signals don't move it).
    floor_t = np.nanmedian(P, axis=1)
    # Drift: robust slope via first-vs-last decile of the session.
    n = len(floor_t)
    k = max(1, n // 10)
    drift = float(np.nanmedian(floor_t[-k:]) - np.nanmedian(floor_t[:k]))

    occupancy = np.nanmean(active, axis=0)          # fraction of time each bin active

    # Cluster contiguous active bins (occupancy above a floor) into emitters.
    of_interest = occupancy > 0.01
    emitters = []
    step = freqs[1] - freqs[0]
    i = 0
    N = len(freqs)
    while i < N:
        if not of_interest[i]:
            i += 1
            continue
        j = i
        run_end = i
        # extend across small gaps (<=2 bins) of inactivity
        while j < N and (of_interest[j] or (j - run_end) <= 2):
            if of_interest[j]:
                run_end = j
            j += 1
        lo, hi = i, run_end
        sub = P[:, lo:hi + 1]
        sub_active = active[:, lo:hi + 1]
        present = sub_active.any(axis=1)            # sweeps where this cluster fired
        occ = float(present.mean())
        peak = float(np.nanmax(sub))
        # mean level while active
        act_vals = sub[sub_active]
        mean_active = float(np.nanmean(act_vals)) if act_vals.size else float("nan")
        present_t = times[present]
        emitters.append({
            "f_lo": float(freqs[lo] - step / 2), "f_hi": float(freqs[hi] + step / 2),
            "f_center": float((freqs[lo] + freqs[hi]) / 2),
            "bw": float((hi - lo + 1) * step),
            "occ": occ, "peak": peak, "mean_active": mean_active,
            "baseline": float(np.nanmedian(baseline[lo:hi + 1])),
            "first": float(present_t.min()) if present_t.size else float("nan"),
            "last": float(present_t.max()) if present_t.size else float("nan"),
            "kind": ("persistent" if occ > 0.9 else "intermittent" if occ > 0.2 else "bursty"),
        })
        i = j
    emitters.sort(key=lambda e: (e["peak"], e["occ"]), reverse=True)
    return {"baseline": baseline, "thresh": thresh, "active": active,
            "floor_t": floor_t, "drift": drift, "occupancy": occupancy,
            "emitters": emitters}


def mhz(hz):
    return f"{hz/1e6:.4f}"


def write_report(out_path, src_name, times, freqs, P, a, margin, floor_pct):
    dur = times[-1] - times[0] if len(times) > 1 else 0.0
    cadence = dur / (len(times) - 1) if len(times) > 1 else 0.0
    ts0 = datetime.fromtimestamp(times[0]).strftime("%Y-%m-%d %H:%M:%S")
    ts1 = datetime.fromtimestamp(times[-1]).strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append(f"# SDR sweep report — {src_name}\n")
    L.append(f"- **Range:** {mhz(freqs[0])}–{mhz(freqs[-1])} MHz, "
             f"{(freqs[1]-freqs[0])/1e3:.3f} kHz bins ({len(freqs)} bins)")
    L.append(f"- **Window:** {ts0} → {ts1}  ({dur:.0f} s, {len(times)} sweeps, "
             f"~{cadence:.2f} s/sweep)")
    L.append(f"- **Detection:** active = bin > per-bin baseline (p{floor_pct}) + "
             f"{margin} dB. Power is uncalibrated dB.\n")

    L.append("## Noise floor")
    drift = a["drift"]
    arrow = "▲ rising" if drift > 1 else "▼ falling" if drift < -1 else "≈ stable"
    L.append(f"- Median floor: {np.nanmedian(a['floor_t']):.1f} dB "
             f"(min {np.nanmin(a['floor_t']):.1f}, max {np.nanmax(a['floor_t']):.1f})")
    L.append(f"- **Drift over session: {drift:+.1f} dB — {arrow}**")
    if abs(drift) > 3:
        L.append(f"  - ⚠ floor moved {abs(drift):.1f} dB — check for creeping RFI / gain/AGC / thermal.")
    band_occ = float((a["occupancy"] > 0.05).mean())
    L.append(f"- Band occupancy: {band_occ*100:.1f}% of bins active >5% of the time\n")

    L.append(f"## Emitters ({len(a['emitters'])} clusters)")
    if not a["emitters"]:
        L.append("_none above threshold_\n")
    else:
        L.append("| center MHz | span kHz | kind | occ % | peak dB | over floor | seen |")
        L.append("|---|---|---|---|---|---|---|")
        for e in a["emitters"][:40]:
            seen = (f"{datetime.fromtimestamp(e['first']).strftime('%H:%M:%S')}"
                    f"–{datetime.fromtimestamp(e['last']).strftime('%H:%M:%S')}"
                    if e["first"] == e["first"] else "—")
            L.append(f"| {mhz(e['f_center'])} | {e['bw']/1e3:.1f} | {e['kind']} | "
                     f"{e['occ']*100:.0f} | {e['peak']:.1f} | "
                     f"{e['peak']-e['baseline']:+.1f} | {seen} |")
        if len(a["emitters"]) > 40:
            L.append(f"\n_…{len(a['emitters'])-40} more clusters omitted_")
    text = "\n".join(L) + "\n"
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text)
    return text


def plot_waterfall(png, times, freqs, P, a, relative=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for the waterfall plot. Install it with "
            "`pip install 'sdr-sweep[analysis]'`, or pass --no-plot for the text report only."
        ) from exc

    dur = times[-1] - times[0] if len(times) > 1 else 1.0
    # Relative mode shows each bin's excess over its own quiet baseline — this
    # removes the (cosmetic) per-hop floor step and makes activity stand out.
    W = (P - a["baseline"]) if relative else P
    label = "dB over baseline" if relative else "dB (uncal)"
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    vmin = 0.0 if relative else float(np.nanpercentile(P, 5))
    vmax = float(np.nanpercentile(W, 99.5))
    # interpolation="nearest" — the default antialiased filter beats against the
    # time-row count and paints spurious horizontal bands.
    im = ax0.imshow(W, aspect="auto", origin="lower", cmap="viridis",
                    vmin=vmin, vmax=vmax, interpolation="nearest",
                    extent=[freqs[0] / 1e6, freqs[-1] / 1e6, 0, dur])
    ax0.set_ylabel("elapsed (s)")
    ax0.set_title(f"Waterfall — power over time{' (relative to baseline)' if relative else ''}")
    fig.colorbar(im, ax=ax0, label=label, pad=0.01)

    ax1.plot(freqs / 1e6, a["baseline"], lw=0.8, label="baseline (quiet)")
    ax1.plot(freqs / 1e6, np.nanmax(P, axis=0), lw=0.8, alpha=0.8, label="peak-hold")
    ax1.fill_between(freqs / 1e6, a["baseline"], np.nanmax(P, axis=0),
                     where=(a["occupancy"] > 0.2), color="orange", alpha=0.3,
                     label="occ >20%")
    ax1.set_xlabel("MHz")
    ax1.set_ylabel("dB (uncal)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    return png


def main():
    ap = argparse.ArgumentParser(description="Analyse an sdr-sweep CSV log")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("csv", help="rtl_power-format CSV from sdr_sweep.py")
    ap.add_argument("--md", default=None, help="write Markdown report here (also printed)")
    ap.add_argument("--png", default=None, help="waterfall PNG path (default <csv>.png)")
    ap.add_argument("--margin", type=float, default=6.0, help="dB over baseline = active (default 6)")
    ap.add_argument("--floor-pct", type=float, default=25.0, help="percentile for per-bin baseline (default 25)")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG")
    ap.add_argument("--relative", action="store_true",
                    help="waterfall shows dB over each bin's baseline (hides hop seam, highlights activity)")
    args = ap.parse_args()

    times, freqs, P = load(args.csv)
    a = analyse(times, freqs, P, args.margin, args.floor_pct)
    text = write_report(args.md, args.csv, times, freqs, P, a, args.margin, args.floor_pct)
    sys.stdout.write(text)

    if not args.no_plot:
        png = args.png or (args.csv.rsplit(".", 1)[0] + ".png")
        plot_waterfall(png, times, freqs, P, a, relative=args.relative)
        print(f"\n[wrote waterfall: {png}]", file=sys.stderr)


if __name__ == "__main__":
    main()
