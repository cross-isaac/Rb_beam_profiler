# Rb_beam_profiler

Characterization of the Rb(-Ar) atomic beam source via 780 nm absorption
spectroscopy: divergence, flux, capture efficiency, and dopant density.

- **`06_Beam_Profiler/`** — **current analysis code.** A simplified rebuild of
  `05_`: one data format (the acquisition-runner run format), one background
  method, and drift-robust peak finding that locates every ramp side
  independently. The laser wanders across a run, which the old fixed-position
  peak search could not follow; this lifts the profile fit from R² 0.37 to 0.96.
  See its [README](06_Beam_Profiler/README.md).
- **`05_Beam_Profiler/`** — the previous pipeline, kept intact and working.
  Handles three on-disk formats, three background methods, and the legacy
  calibration-trace flux — old results were produced with it, so it stays.
  See its [README](05_Beam_Profiler/README.md).
- **`04_Beam_Characterization/`** — original thesis-era notebooks, raw data,
  and results (thesis Chapters 4–5).
- **`IsaacCross_SeniorThesis_2026.pdf`** — the physics reference.
