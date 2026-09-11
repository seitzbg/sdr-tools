# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0]

First public release.

### Features
- **`sdr-sweep`** — the capture engine. Retunes a USRP B210 across a range in
  hops, FFTs each hop, crops the filter roll-off, and stitches the kept bins
  into one spectrum, repeating over time. Writes an rtl_power-compatible CSV log
  and/or emits one JSON object per sweep to stdout (`--stream`).
- **`sdr-sweep-tui`** — live spectrum, scrolling waterfall, and stats rendered
  from the JSON stream. Runs anywhere; pipe it straight off the radio over SSH.
- **`sdr-sweep-report`** — offline analysis of a CSV log: noise-floor level and
  drift, band occupancy, and intermittent/persistent emitter clustering, as a
  Markdown report plus a PNG waterfall.
- **`sdr-sweep-exporter`** — a stdlib-only Prometheus exporter that reads the
  JSON stream and serves `/metrics`. Publishes both latest-sweep gauges and
  rolling-window `_max`/`_mean` companions so bursts and ADC overloads are not
  lost between scrapes.
- **`sdr-sweep-gaincal`** — on-demand RX gain calibration: sweeps gain measuring
  the noise floor and the ADC clip fraction, and reports the sensitivity knee,
  the overload onset, and a recommended gain.
- **Grafana dashboard** (`dashboards/sdr-sweep.grafana.json`) — importable, with
  a `${DS_PROMETHEUS}` datasource variable. Health/overview, noise-floor drift,
  occupancy, peak level + SNR, per-band peak/occupancy, and a per-band occupancy
  state-timeline.

### Design notes (durable)
- **Uncalibrated dB.** Power is relative, not dBm — correct for noise-floor
  drift, occupancy, and burst detection, which are about contrast and change.
- **Antenna.** UHD exposes `TX/RX` and `RX2`; the board's silkscreen "RX1"
  connector is UHD `RX2`, hence `--ant RX2` is the default. RX only.
- **`num_done` burst capture.** Each hop tunes with the stream stopped, then
  issues a `num_done` burst for exactly the samples needed, so there is no
  `recv_num_samps` drain (whose final `recv()` eats the full timeout). This is
  ~26x the RF duty cycle of the drain path, with identical spectral output.
- **Windowed metrics.** A parked band produces far more sweeps than Prometheus
  scrapes, so the latest-sweep gauges discard almost every sweep. The `_max` /
  `_mean` series aggregate every sweep in a rolling window (`--window`, default
  60s) — `sdr_sweep_clip_fraction_max` is the one to alert on for ADC overload.
- **Clip fraction** (`|IQ| >= 0.99`) is a direct, signal-agnostic ADC-overload
  indicator, far more reliable than peak levels (which intermittent signals
  confound).
