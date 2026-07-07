# 04_Beam_Characterization — Absorption spectroscopy of the Rb(-Ar) beam (thesis Ch. 4–5)

Characterization of the beam source via 780 nm absorption spectroscopy on the 85Rb D2
F=3→F′=4 line: spatial profiles at two zones (Zone 1 ≈ 2.46", Zone 2 ≈ 12.3" from nozzle),
divergence (Methods A and B), flux, and capture efficiency.

**Headline results (thesis Table 5.1):** θ½ ranges 0.62–2.1° depending on method/fit
(mean 1.4 ± 0.6°); Rb flux ~2.5–3.2×10^10 atoms/s (≈100× below vapor-pressure expectation
— unresolved); capture efficiency 18–41%; Rb-Ar dopant density ~1–2×10^17 cm^-3.

## Analysis code (this folder)

- **`master_document.ipynb`** — main analysis pipeline (background subtraction → Lorentzian
  fits → spatial profile → divergence/flux). Start here.
- **`beam_analysis.py`** — pipeline functions used by the notebooks.
- **`final_thesis_divergence_plots.ipynb`** — produces the final thesis summary plots.
- **`manual_divergence_fitting.ipynb`, `testing_divergence_fitting_methods.ipynb`,
  `RelativeAbsFitterPlus_Old.ipynb`, `testing.ipynb`** — development/older versions.
- **`ArFlowRate.nb`, `OpticalPowerRbAbsorption.nb`, `og_design_div.nb`,
  `Absorption Calcs (back envelope).nb`** — supporting Mathematica calcs.
- **`Rb_signal_FilterAmplification_circuit.png`** — schematic of the two-stage high-pass +
  op-amp signal conditioning circuit (fc ≈ 15.9 Hz, G ≈ 68×; thesis Sec. 4.3).
- **`helper_scripts/`** — general utilities (`Combine Photos.ipynb`, `PlotandConvert.ipynb`).

## Data and results

- **`absorption_data/`** *(git-ignored — several GB of raw scope CSVs)*
  - `data_101225/` — pure-Rb runs, Oct 12 2025: `Near/` (Zone 1) and `Far/` (Zone 2),
    files named by probe translation-stage position (e.g. `450.csv`); `full*.csv` are
    unattenuated flux-calibration traces; `Near/Discard Data/` = rejected runs.
  - `data_112825/` — dual-species Rb-Ar run, Nov 28 2025 (Zone 1 + `flux_data/`).
  - `background/` — background traces.
- **`results/`** — fit outputs: `results_101225/` and `background_deg1|deg2/` contain
  per-position Lorentzian fit plots, `divergence_results.csv`, `flux_results.csv`;
  `figures/all_versions/` holds the beam-summary PNGs (filenames encode dataset, zone,
  fit model, χ², θ½).

## 780nm_Probe_System/

Probe laser (recommissioned Toptica DL Pro, ~20 mW) beam characterization:
- `780_Beam_Size/` — ThorCam images + `beam_width_fit.py` Gaussian fits giving
  w = 0.275 mm (thesis Fig. 4.7).
- `Beam_size_fitter_thing/` — earlier version of the same fits (WebPlotDigitizer exports).
- `Math/` — `Beam Characterization (DL pro).nb`, `Rb Cell Calcs.nb` (reference cell).

⚠ Path note: notebooks were written when this folder was named `Rb_beam_characterization/`
at the repo root. Internal relative paths (`absorption_data/...`, `results/...`) still work
when running from this folder; only absolute paths, if any, would need updating.
