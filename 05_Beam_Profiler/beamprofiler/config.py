"""
Central configuration for the beam-profiler pipeline.

Every knob in the analysis lives in one dataclass, `AnalysisConfig`, so that a
run is fully reproducible: the pipeline writes the exact configuration used to
`config_used.json` next to the results, and the GUI/CLI can load and save the
same file.

Sections:
  * data layout      — CSV column names and stage-position unit conversion
  * segmentation     — how a raw trace is split into individual ramp sides
  * ramp removal     — polynomial background subtraction settings
  * peak fitting     — Lorentzian fit + noise-rejection thresholds
  * spatial profile  — profile models, error treatment, geometry
  * flux             — physical constants and flux-measurement settings
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AnalysisConfig:
    # ── data layout ───────────────────────────────────────────────────────────
    time_column: str = "Time(s)"          # scope time column
    signal_column: str = "CH2(V)"         # photodetector signal column
    sync_column: str = "CH4(V)"           # square-wave ramp sync from scan controller
    # Probe positions are encoded in the CSV filenames (e.g. "450.csv") in
    # translation-stage units. 39.3701 thou = 1 mm.
    position_units_per_mm: float = 39.3701

    # ── ramp segmentation ─────────────────────────────────────────────────────
    # "sync"   — split the trace at edges of the sync square wave (preferred;
    #            each high/low level is one half of the sawtooth sweep)
    # "auto"   — no sync channel: estimate the ramp turning points from the
    #            heavily-smoothed signal itself (heuristic fallback)
    # "manual" — user supplies explicit side boundaries in `manual_boundaries_s`
    segmentation: str = "sync"
    manual_boundaries_s: list = field(default_factory=list)  # e.g. [-3e-3, -1e-3, 1e-3, 3e-3, 5e-3]
    n_sides_expected: int = 4       # warn if segmentation finds a different count
    edge_trim_fraction: float = 0.08  # fraction trimmed from each side end (ramp turnaround artifacts)
    downsample_points_per_side: int = 3000  # bin-average each side to this many points

    # ── ramp / background removal ────────────────────────────────────────────
    ramp_degree: int = 2            # polynomial degree (2 recommended: RC filter phase
    #                                 shift adds curvature on top of the linear ramp; use 1
    #                                 for unfiltered traces if preferred)
    clip_sigma: float = 2.5         # sigma-clipping threshold for masking the absorption dip
    clip_iterations: int = 6        # max sigma-clip iterations when isolating the background

    # ── Lorentzian peak fitting & noise rejection ────────────────────────────
    fit_peak_offset: bool = True    # allow a constant offset in the peak fit
    # After the reference (strongest) position is fit, other positions are
    # constrained to physically-motivated bounds around the reference values:
    gamma_rel_bounds: tuple = (0.4, 2.5)   # allowed linewidth range, relative to reference Γ
    center_tolerance_frac: float = 0.20    # allowed peak-center wander, as a fraction of
    #                                        the side duration, around the reference center
    # A fitted peak is accepted only if ALL of these hold:
    snr_min: float = 3.0            # fitted amplitude / background-noise RMS
    delta_aic_min: float = 6.0      # AIC improvement of peak model over background-only model
    min_good_sides: int = 2         # position rejected if fewer sides pass validation
    # What to do with rejected positions in the spatial-profile fit:
    #   "zero"    — include as amplitude 0 with a noise-derived upper-bound error (default)
    #   "exclude" — drop from the profile fit entirely
    #   "keep"    — keep the fitted value, but flagged
    rejected_mode: str = "zero"

    # ── spatial profile fitting ──────────────────────────────────────────────
    profile_models: list = field(default_factory=lambda: ["gaussian", "lorentzian"])
    profile_offset: bool = False    # allow a constant offset in the profile fit
    error_mode: str = "STD"         # "STD" (systematic side-to-side scatter) or "SEM"

    # ── geometry (inches, matching the thesis) ───────────────────────────────
    nozzle_diameter_in: float = 0.1     # inscribed-circle diameter of capillary array
    z_probe_in: float = 2.46            # nozzle -> probe distance (Zone 1 = 2.46, Zone 2 = 12.3)
    z_probe2_in: float = 12.3           # nozzle -> second zone (Method B, if a Zone 2 dataset is given)
    z_substrate_in: float = 19.685      # nozzle -> growth substrate
    aperture_radius_cm: float = 0.5     # capture-efficiency aperture half-width

    # ── flux measurement ─────────────────────────────────────────────────────
    oven_temp_C: float = 230.0          # nozzle temperature (sets mean atomic speed)
    sigma_eff_cm2: float = 1.246e-9     # resonant cross-section, isotropic pol, 85Rb D2 F=3->F'=4
    f_F3: float = 7.0 / 12.0            # thermal ground-state fraction in F=3 (degeneracy 7/12)
    p_Rb85: float = 0.7217              # 85Rb natural abundance
    # Stage position (filename units) at which the unattenuated flux trace was
    # taken. None -> parsed from the flux filename (first number found), falling
    # back to the profile-fit center.
    flux_reference_position: float | None = None
    # Ar flow in SCCM for dopant-density estimate; None skips that calculation.
    ar_flow_sccm: float | None = None
    ar_density_crystal_kg_m3: float = 1768.0  # solid Ar density for dopant-density calc

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Write this configuration to a JSON file."""
        d = asdict(self)
        # tuples aren't JSON-native; store as lists
        d["gamma_rel_bounds"] = list(d["gamma_rel_bounds"])
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        """Load a configuration from a JSON file (unknown keys are ignored)."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in valid}
        if "gamma_rel_bounds" in d:
            d["gamma_rel_bounds"] = tuple(d["gamma_rel_bounds"])
        return cls(**d)
