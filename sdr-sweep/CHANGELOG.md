# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.1]

Correctness and reliability fixes across all five tools, from a full code
review. No new commands or flags; two small, backward-compatible format
extensions (noted below). Adds the first automated test suite.

### Fixed
- **Persistent-emitter detection.** The per-bin temporal-percentile baseline made
  a steady carrier its own noise floor, so persistent transmitters vanished from
  occupancy and the emitter list. Detection now takes the lower of the temporal
  baseline and a spectral floor read from each bin's frequency neighbourhood, so
  persistent and intermittent carriers are both found. (`report`)
- **Gain calibration left the exporter stopped** if interrupted during startup:
  the service was stopped *before* the `try`/`finally` that restarts it. The stop
  now happens inside the cleanup scope, stop/restart return codes are checked and
  surfaced, and options + the UHD import are validated before monitoring is taken
  offline. (`gaincal`)
- **Calibration receive could hang forever.** Replaced UHD's retry-until-complete
  `recv_num_samps()` with an explicit deadline-bounded burst receive, so a stalled
  receiver aborts loudly instead of hanging with the service left stopped. (`gaincal`)
- **Calibration recommended a clipping gain** when every tested gain overloaded.
  It now reports no clean operating point (`recommended: null`,
  `clean_gain_exists: false`) and advises lowering gain / adding attenuation. (`gaincal`)
- **Half-bin frequency error.** Stream and CSV bin frequencies were shifted up by
  half an FFT bin; peaks now sit on the true FFT grid across the TUI, exporter,
  report, and dashboard. (`sweep`, `report`)
- **Ignored device sample rate.** Capture and calibration now read `get_rx_rate()`
  back after setting it and rebuild the frequency plan on the actual (possibly
  UHD-coerced) rate. (`sweep`, `gaincal`)
- **Duplicate Prometheus series** for narrow bands: sub-bands whose rounded
  frequency labels collided now carry a stable `idx` label, so each is a distinct
  series and none are silently merged. (`exporter`)
- **TUI crash** on captures with fewer bins than terminal columns — the display
  pooler now always returns the render width. (`tui`)
- **Degenerate capture parameters** (zero bin/hop-bw, `crop = 1`, non-positive
  frames, sub-bin spans) are rejected with a clear message instead of a
  `ZeroDivisionError` or an empty-output crash; the same validation guards
  calibration. (`sweep`, `gaincal`)
- **Run overran its `--duration`** when `--interval` was set: timing now uses a
  monotonic run deadline, checks it before each sweep, and bounds the inter-sweep
  wait by the time left in the run. (`sweep`)
- **CSV out-of-range bins.** CSV output is now trimmed to `--start`/`--stop`, like
  the JSON stream, so a report built from the CSV analyses the requested range. (`sweep`)
- **Mixed bin configurations in one CSV** are rejected with an explanatory error
  instead of crashing or mis-binning old samples. (`report`)
- **Missing observations counted as inactive.** Occupancy now divides active
  observations by *valid* observations; never-observed bins stay unknown rather
  than reading as 0%. (`report`)
- **Dashboard max-SNR combined unrelated sweeps** (`peak_db_max −
  noise_floor_db_mean`). The exporter now publishes a per-sweep
  `sdr_sweep_snr_db` / `sdr_sweep_snr_db_max`, and the dashboard uses it. (`exporter`, dashboard)
- **Malformed stream records** (`null`, a bare list, a null/empty/non-numeric
  `db`) no longer terminate the exporter or TUI; bad lines are skipped and the
  stream continues. (`exporter`, `tui`, new shared `record` validator)
- **Sub-second sweep timing lost** in CSV logs: the time field now carries
  microseconds, so fast runs no longer report zero duration. Legacy whole-second
  logs still load. (`sweep`, `report`)
- **Peak-hold grew instead of decaying** in the TUI: negative-dB levels were
  multiplied by 0.995 (moving them toward 0); the hold now decays by subtracting
  a fixed dB per update. (`tui`)

### Changed
- CSV logs gain two backward-compatible extensions — microsecond timestamps and
  trimming to the requested range. `sdr-sweep-report` reads old and new logs.
- Per-band Prometheus series now carry an additional `idx` label; existing band
  queries keep working, but the series identity changes once at upgrade.
- Added a `pytest` suite (`tests/`) and a `test` optional-dependency group; CI now
  installs the wheel and runs the tests plus CLI smoke checks.

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
