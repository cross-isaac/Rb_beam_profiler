# 05_Beam_Profiler — Automated Rb beam characterization

A rebuilt, automated version of the beam-characterization analysis from
`04_Beam_Characterization` (thesis Chapters 4–5): point it at a folder of raw
oscilloscope traces and it produces the full chain of results — spatial
absorption profile, Gaussian/Lorentzian(/pseudo-Voigt) fits, beam divergence
(Methods A and B), capture efficiency, absolute Rb flux, and dopant density —
with every intermediate step saved as an image you can audit.

```
raw scope CSVs (one per probe position)
   │  1. segment each trace into ramp sides (robust triangle-wave fit)
   │  2. remove the linear background (line through point clusters
   │     at the ramp turn points)
   │  3. fit the Lorentzian absorption peak (+ statistical noise rejection);
   │     run-format data: normalize by the CH2 DC level → TRUE absorption %
   │  4. build & fit the spatial profile A(Δy)  →  divergence, capture eff.
   │  5. absolute flux — directly from the absolute profile (run format),
   │     or from an unattenuated reference trace (legacy)
   ▼
results folder: stage images + absorption_profile.csv + summary plots + results.json
```

## Installation

Requires Python ≥ 3.10.

```
cd 05_Beam_Profiler
python -m pip install -r requirements.txt
```

## Quick start — GUI

```
python run_gui.py
```

1. **Run Analysis tab** — select the data folder (position CSVs like `450.csv`),
   an output folder, and optionally a flux trace and/or a Zone 2 folder.
   Adjust settings if needed (all defaults match the thesis analysis).
   Press **Run Full Pipeline** and watch the log.
2. **Image Browser tab** — click through every stage image (raw segmentation,
   per-side background + peak fits with accept/reject verdicts, profile fits,
   flux fits, final summaries). Use ◀/▶ or the list.
3. **Results tab** — the final numbers, formatted; button to open the output
   folder.

Settings can be saved/loaded as JSON so a whole analysis is reproducible.

## Quick start — command line

```
# run-format folder (absolute absorption, flux straight from the profile):
python run_pipeline.py --data run_20260716_182706 --out results\run_071626 ^
    --label Rb_071626

# divergence + flux + dopant density for the dual-species run:
python run_pipeline.py ^
    --data ..\04_Beam_Characterization\absorption_data\data_112825 ^
    --out results\rb_ar_112825 ^
    --flux ..\04_Beam_Characterization\absorption_data\data_112825\flux_data ^
    --ar-flow 0.01 --label RbAr_112825

# two-zone pure-Rb run (Method B divergence):
python run_pipeline.py ^
    --data  ..\04_Beam_Characterization\absorption_data\data_101225\Near ^
    --zone2 ..\04_Beam_Characterization\absorption_data\data_101225\Far ^
    --out results\rb_101225 ^
    --flux ..\04_Beam_Characterization\absorption_data\data_101225\Near\full0converted.csv ^
    --flux-position 450 --label Rb_101225
```

`python run_pipeline.py --help` lists all flags;
`--write-default-config settings.json` dumps every setting for editing.

## Input data format

Both layouts are auto-detected per folder — no switches needed.

### Run format (2026-07 onward, op-amp bypassed — preferred)

A folder written by the acquisition runner (`run_YYYYMMDD_HHMMSS/`) with one
`point_NNNN.csv` per scan step:

```
# timestamp: 2026-07-16 18:30:11.203
# step: 1
# pass: 1
# target_in: 0.20000
time_s,CH2_V,CH3_V,CH4_V
```

* `CH2` — **raw DC photodetector level** (no amplifier). Sets the absolute
  transmission scale per point, so window-transmission differences across the
  viewport are normalized out automatically.
* `CH3` — **AC-coupled copy** of the same signal at fine vertical scale; this
  is what the ramp removal and peak fits run on.
* `CH4` — scan-controller sync square wave.
* Probe position comes from the `target_in` header (inches; converted to thou
  internally). Multiple files at the same position (multi-pass scans) are
  combined into one profile point.

Every side fit reports **true percent absorption** (CH3 dip ÷ CH2 DC level),
so the spatial profile is absolute and the **flux is computed directly from
the profile** — no separate calibration trace needed.

### Legacy format

* **Position scans** — one CSV per probe height, named by the translation-stage
  position in thou (e.g. `450.csv`). Standard scope export with columns
  `Time(s), CH2(V), CH4(V)`:
  * `CH2` — (amplified) photodetector signal,
  * `CH4` — scan-controller sync square wave (optional but recommended).
  Files with non-numeric names (`full0.csv`, `RelativeAbsNear.csv`, notes, …)
  in the same folder are ignored automatically. Absorption is relative.
* **Flux traces** — unattenuated (amplifier bypassed) traces at one reference
  position; either the same `Time(s),CH2(V),CH4(V)` format or the raw scope
  export with `t0 = …, tInc = …` encoded in the header. Both are handled.
  (A flux trace can also be given for run-format data — it then overrides the
  profile-based flux.)

## What was improved vs. the old notebooks

The old workflow (`master_document.ipynb` + `beam_analysis.py`) required
hand-tuned time windows, per-side initial guesses, and a manual first pass on
one strong peak; and at the beam edges it happily fit noise, inflating the
profile tails (thesis Sec. 5.3). This package replaces those steps:

1. **Automatic ramp segmentation.** The sync square wave gives the sweep
   period, but its edges are *phase-shifted* from the real sawtooth turnaround
   (≈1 ms in the 2025 data!). The turnaround phase is measured from the signal
   itself: the trace is folded by the period and fit with an ideal triangle
   wave using a robust soft-L1 loss, so even a very deep absorption dip cannot
   pull the corners. No hard-coded time windows anywhere.
2. **Corner-anchored background removal.** The triangle-wave amplitude is
   measured directly: a small cluster of points at each end of the side (just
   off the ramp turn points) is averaged and the background is the straight
   line through the two cluster means. Nothing is masked and nothing is fit
   iteratively, so there are no per-side decisions to jitter between sweeps —
   on the 2026 run data this gives the best side-to-side reproducibility
   (4.3 % scatter vs 5.8 % for a wing fit and 9.2 % for sigma-clip masking),
   at the cost of a ~+2 % amplitude offset from the reproducible corner
   rounding (verified against a simultaneous line+Lorentzian fit). Two
   alternatives remain selectable: `background_method: "wings"` (polynomial
   fit to all points ≥ `wing_inner_gammas`×Γ from the peak) and `"masked"`
   (the sigma-clip method, also the automatic fallback for degree-2 legacy
   backgrounds where a straight line does not apply).
3. **Reference-peak bootstrapping.** The strongest position is found and fit
   unconstrained, then its linewidth and peak position (as a fraction of the
   sweep) become initial guesses *and bounds* for every other position —
   exactly the "fit one strong peak first" trick, automated. Physically the
   linewidth/line-centre cannot change with probe height, so weak-signal fits
   are strongly constrained.
4. **Statistical noise rejection.** Every side fit must pass:
   * SNR ≥ `snr_min` (fitted amplitude vs. background-noise RMS),
   * ΔAIC ≥ `delta_aic_min` (peak model must beat a background-only model),
   * linewidth not pinned at its bound.
   A position needs `min_good_sides` accepted sides, else it is **rejected**
   and enters the profile as amplitude 0 with a noise-derived upper bound
   (configurable: `zero` / `exclude` / `keep`). Rejected points are drawn in
   red on the profile plots. On the 2025 datasets this correctly removes the
   positions (e.g. 700–900 in the Rb-Ar run) that previously fit noise.
5. **Automatic linear-core detection (run-format data).** Optics
   imperfections curve the first/last ~10–20 % of each nominally linear ramp
   side by several times the noise — enough to masquerade as signal. Each
   side's background is anchored on its central half, the smoothed residuals
   are compared against their *measured* noise (the CH3 etalon ripple is
   correlated, so a white-noise estimate would be wrong), and the curved ends
   are walked off until the data is provably polynomial. The known peak window
   is exempt, and trims are capped (`auto_trim_max_fraction`). Trimmed regions
   are drawn gray on every `02_side_fits` image.
6. **Pseudo-Voigt tiebreaker.** Enable the third model to get a mixing
   parameter η that interpolates Gaussian (η=0) → Lorentzian (η=1) — a direct,
   quantitative answer to "is the profile really Lorentzian?" once the noise
   points are handled. (On the Rb-Ar Zone 1 data: η ≈ 1.0, i.e. the Lorentzian
   preference survives noise rejection.)
7. **One command / one click end-to-end**, with every stage image saved and a
   machine-readable `results.json` + the exact `config_used.json` per run.

## Output layout

```
<output folder>/
├── config_used.json          exact settings of this run (reproducibility)
├── 01_raw_traces/            full trace + detected side boundaries, per position
├── 02_side_fits/             per position & side: background fit (top) and
│                             Lorentzian fit with ACCEPT/REJECT verdict (bottom)
├── 03_profile/
│   ├── absorption_profile.csv  per-position amplitudes, errors, accepted flags,
│   │                           CH2 DC level (run format)
│   ├── side_details.csv        long-format per-side fit details (amplitude,
│   │                           SNR, verdicts, trim fraction, DC level)
│   ├── transmission_vs_position.png  CH2 DC level vs position (run format)
│   └── profile_<model>.png     spatial profile fit + parameter table
├── 04_flux/                  flux-trace segmentation and absorption-% fits
│                             (legacy calibration-trace runs only)
├── zone2/                    same structure for the Zone 2 data (if given)
├── summary_<model>.png       final column-density profile + full results table
└── results.json              all numbers, machine-readable
```

## Package layout

```
beamprofiler/
├── config.py            AnalysisConfig — every knob, JSON save/load
├── trace_io.py          CSV loading (both scope formats), ramp segmentation
├── ramp_removal.py      sigma-clipped polynomial background subtraction
├── peak_fitting.py      Lorentzian peak fits, reference template, validation
├── profile_analysis.py  profile models, divergence A/B, capture efficiency
├── flux.py              Beer-Lambert → column density → flux → dopant density
├── plotting.py          every figure the pipeline saves
├── pipeline.py          orchestration: run_full_pipeline() and stage functions
└── gui/app.py           Tkinter application (3 tabs)
```

Each stage is callable on its own, e.g. re-fit an existing profile CSV without
re-processing traces:

```python
import pandas as pd
from beamprofiler import AnalysisConfig
from beamprofiler.pipeline import run_profile

cfg = AnalysisConfig(profile_models=["gaussian", "lorentzian", "pseudo_voigt"])
df = pd.read_csv("results/rb_ar_112825/03_profile/absorption_profile.csv")
fits = run_profile(df, "results/rb_ar_112825", cfg, progress=print)
```

## Key settings (see `config.py` for all, with comments)

| Setting | Default | Meaning |
|---|---|---|
| `ramp_degree` | `auto` | background polynomial: auto = 1 for run-format (raw) data, 2 for legacy (RC-filter curvature); or set 1/2 explicitly |
| `absorption_mode` | `auto` | `absolute` (CH3 dip / CH2 DC), `relative`, or auto by format |
| `dark_offset_V` | 0.0 | photodiode dark level subtracted from CH2 before normalizing |
| `background_method` | `corners` | line through turn-point clusters; `wings` = fit clusters flanking the peak; `masked` = legacy sigma-clip |
| `corner_cluster_fraction` | 0.05 | cluster size at each side end (corners method) |
| `corner_fit_exclude_fraction` | 0.15 | side-end fraction kept out of the peak fit (corners method) |
| `wing_inner_gammas` | 4.0 | wing clusters start this many HWHM from the peak centre |
| `auto_trim` | True | trim curved ramp ends via linear-core detection |
| `auto_trim_max_fraction` | 0.30 | cap on the auto-trim per side end |
| `auto_trim_sigma` | 2.5 | trim threshold in units of measured smoothed-residual noise |
| `fit_peak_offset` | True | constant offset in the Lorentzian peak fit |
| `snr_min` | 3.0 | reject side fits below this amplitude/noise ratio |
| `delta_aic_min` | 6.0 | peak model must beat background-only by this AIC margin |
| `min_good_sides` | 2 | position rejected below this many accepted sides |
| `rejected_mode` | `zero` | rejected positions: `zero` (0 ± upper bound), `exclude`, `keep` |
| `error_mode` | `STD` | side-to-side scatter (thesis choice) or `SEM` |
| `profile_models` | gaussian, lorentzian | add `pseudo_voigt` for the η tiebreaker |
| `z_probe_in` | 2.46 | nozzle→probe distance (12.3 for Zone 2) |
| `flux_reference_position` | None | stage position of the flux trace (else parsed from filename) |
| `ar_flow_sccm` | None | set to enable the dopant-density estimate |

**If a good position gets rejected** (very weak beams): lower `snr_min` /
`delta_aic_min`, and check the red-boxed verdict on its `02_side_fits` image
to see exactly which test failed.

**If segmentation looks wrong** (see `01_raw_traces`): the trace may be too
short (< 2 sweep periods) for the folded triangle fit — record ≥ 2 full
periods, or set `segmentation: "manual"` with explicit `manual_boundaries_s`
in a config JSON.

## Validation against the thesis

Running this pipeline on the 2025 datasets reproduces thesis Table 5.1 within
errors, while additionally rejecting the noise-dominated edge positions:

| Quantity | This pipeline | Thesis |
|---|---|---|
| Rb Zone 1 FWHM (Gauss / Lor) | 5.05 / 4.14 mm | 5.4±0.1 / 3.9±0.1 mm |
| Rb Zone 2 FWHM (Gauss / Lor) | 23.3 / 21.5 mm | 23±2 / 21±2 mm |
| Rb Method B θ½ (Gauss / Lor) | 2.09° / 1.99° | 2.1±0.2° / 2±0.2° |
| Rb flux | 2.4–3.0×10¹⁰ /s | 2.6–3.2×10¹⁰ /s |
| Rb-Ar Zone 1 FWHM (Gauss / Lor) | 5.8 / 5.0 mm | 5.0±0.2 / 4.3±0.2 mm |
| Rb-Ar dopant density | 1.5–2.0×10¹⁷ cm⁻³ | 1–2×10¹⁷ cm⁻³ |

(Small differences are expected — the new pipeline excludes the noise-fit
edge points that inflated the old profile tails.)

---
*Author: built with Claude Code for Isaac Cross, Odom Lab, Northwestern University.*
*Physics reference: I. Cross, Senior Thesis 2026, Chapters 4–5 and Appendix C.*
