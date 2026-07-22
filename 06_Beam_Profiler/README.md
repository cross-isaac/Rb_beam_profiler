# 06_Beam_Profiler — Rb beam characterization (run-format, simplified)

A rebuilt, streamlined version of `05_Beam_Profiler`. It reads **one** data
format (the acquisition-runner "run" format) and produces the full chain of
results — absolute spatial absorption profile, Gaussian/Lorentzian(/pseudo-Voigt)
fits, beam divergence (Method A), capture efficiency, absolute Rb flux, and
optional dopant density — with every intermediate step saved as an image.

```
run-format scope CSVs (one point_NNNN.csv per probe position)
   │  1. segment every trace into ramp sides — one whole-run triangle-wave fit
   │     fixes the sweep period/duty; each trace contributes only its phase
   │  2. per side: peak-aware corners background (line through the baseline
   │     nearest each end, excluding the located peak)
   │  3. per side: locate the absorption dip with a matched filter (whole-side
   │     search — robust to laser drift), fit a Lorentzian, validate (SNR + AIC)
   │  4. combine sides -> absolute profile A(Δy); fit Gaussian/Lorentzian
   │  5. absolute flux straight from the profile (CH3 dip / CH2 DC level)
   ▼
results/: stage images + absorption_profile.csv + summary plots + results.json
```

## Why this rebuild (vs. `05_Beam_Profiler`)

The old package handled three on-disk formats, three background methods, two
absorption modes, a legacy calibration-trace flux path, and two-zone Method B —
almost none of which the current apparatus produces. More importantly, its
peak finder **locked every side's peak position to a single reference position**
(±20 % of the sweep). The laser drifts across a run — and is nudged back by hand
mid-scan — so the resonance wanders over almost the whole sweep (measured:
fractional position 0.23–0.87). Drifted peaks fell outside the locked window, so
they were fit to the wrong place or missed entirely; on the test dataset this
zeroed the **centre** of the beam and gave a profile R² of 0.37.

This version:

1. **One data format.** Run format only (`point_NNNN.csv`, `# target_in`
   header, columns `time_s,CH2_V,CH3_V,CH4_V`). No format auto-detection.
2. **One background method — peak-aware corners.** The background under each
   ramp side is the straight laser-power ramp, pinned by the baseline points
   nearest each *end* of the side (the "corners"). Those anchor clusters exclude
   the located peak, and a cluster on the peak side slides inward to clean
   baseline, so a dip sitting near a ramp apex never tilts the line.
   `corner_cluster_fraction` sets how much of each edge is used.
3. **Drift-robust peak finding.** Every side is localized *independently* by a
   matched filter (a Lorentzian template of the reference linewidth slid over
   the whole side), then fitted locally. No position is inherited between sides,
   so unpredictable drift is a non-issue. The linewidth is free within generous
   bounds — the line genuinely broadens toward the beam edges — while a fit
   pinned at the broad bound (a runaway absorbing ramp curvature) is rejected.
4. **Side-level rejection.** A side that fails the SNR/AIC test counts as
   0 absorption and is averaged in with the passing sides (`rejected_mode:
   zero`), so a genuine beam-edge position enters the profile honestly. `exclude`
   and `keep` remain available.
5. **Consistent segmentation.** The sweep period and duty are fixed instrument
   settings, so they are fitted per trace, reduced to a run-wide median, and
   applied to every trace (phase-only per trace). One bad per-trace fit can no
   longer mis-segment a file.

On the test dataset the rebuild recovers every previously-lost side (the beam
centre now sits at the profile peak) and lifts the Gaussian profile R² from 0.37
to 0.96.

## Installation

Requires Python ≥ 3.10.

```
cd 06_Beam_Profiler
python -m pip install -r requirements.txt
```

## Quick start — GUI

```
python run_gui.py
```

1. **Run Analysis** — pick the data folder (`point_*.csv`) and an output folder,
   adjust settings if needed, press **Run Full Pipeline**, watch the log. Each
   new run writes to its own `<output folder>/<dataset label>/` subfolder (a
   numeric suffix is added if that name already exists), so runs never overwrite
   each other. A **re-run** from the Image Browser (applying manual verdicts)
   reuses the current run's folder instead of making a new one.
2. **Image Browser** — click through every stage image (segmentation, per-side
   background + peak fits with accept/reject verdicts, profile fits, summaries).
   - **Position filter** narrows the per-side images to one probe position.
   - **Clickable profile:** in the `03_profile` stage the `profile_<model>.png`
     is drawn as a *live* plot — the same profile with its fit curve — and
     clicking a point jumps straight to the `02_side_fits` images behind it. Blue
     = clean, amber diamond = suspect (review), red = noise-dominated; a purple
     ring marks a position that already carries a hand-set verdict, and a green
     ring shows the point you last jumped from. The other `03_profile` diagnostics
     (transmission, sawtooth amplitude) stay as plain images in the same list.
   - **Side verdict bar** forces the selected side to ACCEPT or REJECT when the
     automatic SNR/dAIC/runaway tests got it wrong. Verdicts are saved to
     `manual_overrides.json` and take effect on the next run ("Re-run pipeline"
     is right there); the automatic verdict is always shown alongside so an
     override can never be mistaken for a fit that passed on its own.
3. **Results** — the final numbers; button to open the output folder.

Side-image / list markers: `+` accepted, `-` rejected, `?` suspect (flag mode —
accepted but the auto-tests failed; review it), `M` a hand-set verdict pending a
re-run.

Settings save/load as JSON so a whole analysis is reproducible.

## Quick start — command line

```
python run_pipeline.py --data run_20260720_1642 --out results\run_072026 --label Rb_072026

# with the dopant-density estimate:
python run_pipeline.py --data <folder> --out <results> --ar-flow 0.01
```

`python run_pipeline.py --help` lists the flags;
`--write-default-config settings.json` dumps every setting for editing.

## Input data format

A folder written by the acquisition runner, one `point_NNNN.csv` per scan step:

```
# timestamp: 2026-07-20 16:48:33.944
# step: 4
# target_in: 0.50000
time_s,CH2_V,CH3_V,CH4_V
```

* `CH2` — raw DC photodetector level (no amplifier). Sets the absolute
  transmission scale per side, so window-transmission differences across the
  viewport are normalized out.
* `CH3` — AC-coupled copy of the same signal at fine vertical scale; this is
  what the background removal and peak fits run on.
* `CH4` — scan-controller sync square wave (gives the sweep period).
* Probe position comes from `# target_in` (inches → thou). Multiple files at the
  same position are combined into one profile point.

Every side reports **true percent absorption** (CH3 dip ÷ CH2 DC level), so the
profile is absolute and the flux comes directly from it — no calibration trace.

## Output layout

The GUI writes each run into a `<output folder>/<dataset label>/` subfolder (see
above); the CLI writes straight into `--out`. Either way one run folder holds:

```
<run folder>/
├── config_used.json            exact settings of this run (reproducibility)
├── manual_overrides.json       hand-set per-side verdicts, if any (see below)
├── 01_raw_traces/              full trace + detected side boundaries, per position
├── 02_side_fits/               per side: peak-aware background (top) and
│                               Lorentzian fit with ACCEPT/REJECT verdict (bottom)
├── 03_profile/
│   ├── absorption_profile.csv    per-position amplitude, error, flags, suspect, CH2 level, sawtooth amp
│   ├── side_details.csv          per-side amplitude, centre, HWHM, SNR, verdicts (auto + suspect + manual)
│   ├── transmission_vs_position.png   CH2 DC level vs position
│   ├── amplitude_vs_position.png      CH3 sawtooth (laser-sweep) amplitude vs position
│   └── profile_<model>.png       spatial profile fit + parameter table
├── summary_<model>.png         final column-density profile + full results table
└── results.json                all numbers, machine-readable
```

**`amplitude_vs_position.png`** is the sawtooth-amplitude companion to the
transmission diagnostic: top panel is the CH3 laser-power sweep's peak-to-peak
span [V], measured over the fitted (edge-trimmed) side; bottom panel is that
span ÷ the CH2 DC level, i.e. the fractional sweep depth reaching the detector,
which should be **flat** across the scan if CH3 is simply the AC copy of CH2. A
slope or step there flags a gain/range change the absolute-absorption scale
should not be trusted through.

## Verdict mode: reject vs. flag-for-review

`side_verdict_mode` decides what the automatic SNR / dAIC / runaway tests *do* to
a side that fails them:

- **`auto`** (default) — a failing side is **rejected**. Under the default
  `rejected_mode: zero` a beam-edge position with no resolvable line is counted
  as zero absorption, which is the physically conservative, thesis behaviour.
- **`flag`** — **nothing is rejected automatically.** Every side that produced a
  fit is *accepted*, and a failing one is marked **suspect** (amber in the
  browser, `?` in the list, a `suspect` column in the CSVs). You then eyeball the
  suspects and Force REJECT the genuinely bad ones by hand. Use this to review
  every point yourself instead of trusting the thresholds.

  ⚠ **This changes the numbers.** Until you reject them, noise-only edge sides
  enter the profile at their fitted (noise-level) amplitude, which *broadens the
  fitted beam* — on the TestFitting dataset the Gaussian FWHM goes 6.39 → 8.52 mm
  and capture efficiency drops accordingly. A `flag` run is therefore **not
  comparable to an `auto` run** unless you have reviewed and rejected the
  suspects. In both modes the reference linewidth is still measured only from
  sides that genuinely pass the tests, so switching modes never moves the
  matched-filter template.

The default stays `auto` so previously computed results are unchanged; set
`flag` in the GUI ("Verdict mode"), via `--verdict-mode flag`, or in the config.

## Manual side verdicts

The automatic SNR / dAIC / linewidth-runaway tests are deliberately blunt so
they can run unattended. When a verdict is obviously wrong on the
`02_side_fits` image, override it from the Image Browser instead of loosening a
threshold run-wide (which moves every other side too). Overrides live in
`manual_overrides.json` in the output folder, keyed by the side label
(`<file stem>:s<side>`, the same key as in `side_details.csv`), and are applied
to the **final** fitting pass only — never to the pass that measures the
reference linewidth, so a side kept by hand cannot redefine the matched-filter
template. Every override is auditable: both verdicts land in `side_details.csv`
(`accepted` vs `auto_accepted`), the side image says MANUAL OVERRIDE in a purple
box, the profile/summary figures carry a footnote, and `results.json` records
the full map. Set `use_manual_overrides: false` (or `--no-overrides`) to
reproduce a purely automatic analysis without deleting the file.

## Package layout

```
beamprofiler/
├── config.py            AnalysisConfig — every knob, JSON save/load
├── trace_io.py          run-format CSV loading, whole-run triangle segmentation
├── ramp_removal.py      peak-aware corners background
├── peak_fitting.py      matched-filter localization, Lorentzian fit, validation
├── side_overrides.py    hand-set per-side accept/reject verdicts (manual_overrides.json)
├── profile_analysis.py  profile models, Method A divergence, capture efficiency
├── flux.py              Beer-Lambert → column density → flux → dopant density
├── plotting.py          every figure the pipeline saves
├── pipeline.py          orchestration: run_full_pipeline() and stage functions
└── gui/app.py           Tkinter application (3 tabs)
```

## Key settings (see `config.py` for all, with comments)

| Setting | Default | Meaning |
|---|---|---|
| `corner_cluster_fraction` | 0.06 | how much of each edge anchors the background line |
| `peak_exclude_gammas` | 2.0 | peak half-window (×HWHM) kept out of the background anchor |
| `gamma_rel_bounds` | (0.4, 2.5) | refined-linewidth bounds, ×reference HWHM (allows edge broadening; rejects broad runaways) |
| `snr_min` | 3.0 | reject a side fit below this amplitude/noise ratio |
| `delta_aic_min` | 6.0 | peak model must beat background-only by this AIC margin |
| `linewidth_runaway_fraction` | 0.97 | how close to the linewidth ceiling counts as "ran away" (1.0 = only exactly pinned; 0 = test off) |
| `side_verdict_mode` | `auto` | `auto` rejects failing sides; `flag` accepts every side and marks failures "suspect" for review (see below) |
| `use_manual_overrides` | True | honour hand-set verdicts in `manual_overrides.json` |
| `rejected_mode` | `zero` | failed side: `zero` (0, averaged in), `exclude`, or `keep` |
| `min_good_sides` | 1 | fewer passing sides → position flagged (drawn red) |
| `error_mode` | `STD` | per-position error: `STD` / `SEM` / `upper_bound` |
| `profile_models` | gaussian, lorentzian | add `pseudo_voigt` for the η tiebreaker |
| `z_probe_in` | 2.46 | nozzle→probe distance |
| `ar_flow_sccm` | None | set to enable the dopant-density estimate |

**If a good position gets rejected:** check the red-boxed verdict on its
`02_side_fits` image for which test failed. If it says *SNR* / *dAIC*, lower
`snr_min` / `delta_aic_min`. If it says *linewidth ran away*, either raise
`linewidth_runaway_fraction` toward 1.0 (only reject fits truly pinned at the
bound) or raise the ceiling itself via `gamma_rel_bounds[1]` (GUI: "Max
linewidth"). For a one-off — a single side the tests keep getting wrong — just
force its verdict in the Image Browser rather than moving a threshold for the
whole run.

---
*Author: rebuilt with Claude Code for Isaac Cross, Odom Lab, Northwestern University.*
*Physics reference: I. Cross, Senior Thesis 2026, Chapters 4–5 and Appendix C.*
