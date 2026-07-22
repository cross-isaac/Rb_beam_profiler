# Rb_beam_profiler

Analysis of the Rb(–Ar) atomic beam source via 780 nm absorption spectroscopy:
divergence, flux, capture efficiency, dopant density. **No hardware** — this
repo only reads scope CSVs. See the root `../CLAUDE.md` for the git and physics
rules.

Physics reference: `IsaacCross_SeniorThesis_2026.pdf`, Chapters 4–5 and
Appendix C.

## Which directory to work in

- **`06_Beam_Profiler/`** — the current pipeline. Run format only, one
  background method, drift-robust peak finding. Work here by default.
- **`05_Beam_Profiler/`** — the previous pipeline, kept intact and working.
  Handles three on-disk formats, three background methods, Method B / zone-2,
  and the legacy calibration-trace flux. **Do not delete or "unify" it** — old
  results were produced with it.
- **`04_Beam_Characterization/`** — thesis-era notebooks and results.

```
cd 06_Beam_Profiler
python run_gui.py
python run_pipeline.py --data <folder> --out <results> --label <name>
python run_pipeline.py --write-default-config settings.json   # dump every knob
```

## The bug that shaped 06

v1 locked every side's peak position to a single reference position, ±20% of the
sweep. But **the laser drifts across a run and is nudged back by hand
mid-scan**, so the resonance wanders over fractional position 0.23–0.87 with no
smooth trend. Drifted peaks fell outside the locked window and were fit to the
wrong place or missed — which zeroed the *centre* of the beam and gave a profile
R² of 0.37.

06 localizes every side independently with a matched filter over the whole side.
The reference is used **only** to scale the Lorentzian template's linewidth
(median HWHM of the highest-SNR sides), never as a position lock. R² 0.37 → 0.96.

If you touch peak finding: never reintroduce a position prior between sides.

## Things the data forced (don't undo these)

- The resonance is crossed **near the ramp apex**, so the dip sits near one end
  of each side. Corner anchors must be peak-aware — nearest baseline to each end,
  excluding the located peak — or they tilt the background line.
- A broad Lorentzian with a free offset **can run away and absorb the apex
  curvature**. Guarded by tying fitted HWHM to a robust reference scale and
  rejecting fits pinned at the broad bound.
- Sweep period and duty are fixed instrument settings for a run, so they are
  fitted per trace, reduced to a run-wide median, and applied globally with
  phase-only per trace. One bad per-trace fit would otherwise mis-segment a file.
- The up/down-ramp amplitude asymmetry seen in v1 was a **fitting artifact, not
  physical**.

## Run format

`point_NNNN.csv` with a `# target_in` header and `time_s,CH2_V,CH3_V,CH4_V`.
CH2 is the raw DC photodetector level, CH3 the AC-coupled copy that fits run on,
CH4 the sync square wave. Dividing the CH3 dip by the CH2 DC level gives **true
percent absorption**, so window-transmission differences across the viewport
cancel and flux comes straight from the profile — no calibration trace.

## Status

06's core pipeline is validated on `05_Beam_Profiler/TestFitting`. It has **not**
been run against a fresh acquisition folder, and the GUI has not been exercised
interactively. Say so in anything that reports its numbers.

## Never tracked

`TestFitting/`, `TestFittingResults/`, `Test/test_output*/`, `results/`,
`__pycache__/`. The curated thesis figure in
`04_Beam_Characterization/results/` **is** tracked deliberately — it is an
output kept as a result, not something the code regenerates.
