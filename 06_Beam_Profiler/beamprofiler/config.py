"""
Central configuration for the beam-profiler pipeline (run-format only).

Every knob lives in one dataclass, `AnalysisConfig`, so a run is fully
reproducible: the pipeline writes the exact configuration used to
`config_used.json`, and the GUI/CLI can load and save the same file.

This is the simplified v2 of the analysis. It handles ONE data layout — the
acquisition-runner "run" format (point_NNNN.csv with a `# target_in` header and
columns time_s/CH2_V/CH3_V/CH4_V) — and ONE background method (peak-aware
corners). See the README for the format spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AnalysisConfig:
    # ── data layout ───────────────────────────────────────────────────────────
    # Probe positions come from the "# target_in" header (inches) and are
    # converted to thou on load. 39.3701 thou = 1 mm.
    position_units_per_mm: float = 39.3701
    # Photodiode dark level [V] subtracted from CH2 (the DC transmission level)
    # before it is used to normalize CH3 to an absolute absorption percentage.
    dark_offset_V: float = 0.0

    # ── ramp segmentation ─────────────────────────────────────────────────────
    # The CH4 sync square wave gives the sweep PERIOD; the actual turning-point
    # phase is measured from the signal via a robust triangle-fold fit (the sync
    # edges are phase-shifted from the real sawtooth). Each side = one up- or
    # down-sweep through the resonance.
    n_sides_expected: int = 4          # note (not an error) if a trace differs
    edge_trim_fraction: float = 0.08   # fraction trimmed from each side end (turnaround artifacts)
    min_side_fraction: float = 0.6     # drop sides shorter than this * median side duration
    downsample_points_per_side: int = 3000  # bin-average each side to at most this many points

    # ── background removal: peak-aware corners ────────────────────────────────
    # The background under each side is the linear laser-power ramp. It is
    # removed by drawing a straight line through a small cluster of points at
    # each end of the side ("corners"), i.e. the triangle-wave amplitude is
    # measured directly. The clusters are made PEAK-AWARE: the located
    # absorption window is excluded, and any cluster that would overlap the peak
    # is slid inward off it, so a dip sitting near a ramp apex cannot tilt the
    # background line.
    corner_cluster_fraction: float = 0.06  # width of each end anchor cluster,
    #                                        as a fraction of the side. This is
    #                                        the "how much of the edge is used"
    #                                        knob — raise it for a steadier line
    #                                        on noisy sides, lower it if the
    #                                        ramp ends are curved.
    peak_exclude_gammas: float = 2.0       # exclude +/- this * gamma around the
    #                                        located peak from the anchor clusters
    #                                        (kept modest so an off-centre peak
    #                                        does not push the anchors off the ends)

    # ── peak localization & Lorentzian fit ────────────────────────────────────
    # Each side is fitted INDEPENDENTLY: the peak is first localized by a
    # matched filter (a Lorentzian template of the reference linewidth slid over
    # the whole side), then a local Lorentzian is fitted, seeded there. The
    # laser drifts unpredictably across a run (manual corrections mid-scan), so
    # NOTHING about the peak position is inherited between sides.
    fit_peak_offset: bool = True           # allow a constant offset in the peak fit
    # Matched-filter template widths, as multiples of the reference HWHM. A
    # small bank lets edge-broadened (more divergent / thermal) lines register
    # even though localization uses the reference scale.
    match_widths_rel: tuple = (0.7, 1.0, 1.6, 2.5)
    # Bounds for the refined linewidth, relative to the width scale (reference
    # HWHM, or the half-max estimate in the bootstrap pass). Wide enough for the
    # real broadening toward the beam edges, tight enough that a broad Lorentzian
    # cannot run away and absorb the curved ramp ends.
    gamma_rel_bounds: tuple = (0.4, 2.5)
    center_refine_gammas: float = 4.0      # refined centre may move +/- this * gamma
    #                                        from the matched-filter location
    search_margin_fraction: float = 0.03   # keep the matched-filter search this
    #                                        far from the very ends of the side

    # ── noise rejection (per side) ────────────────────────────────────────────
    # A fitted side is accepted only if BOTH hold:
    snr_min: float = 3.0            # fitted amplitude / background-noise RMS
    delta_aic_min: float = 6.0      # AIC improvement of the peak model over background-only
    # (No "linewidth pinned" test — legitimate edge broadening must not be
    # rejected as a bad fit.)

    # ── combining sides into one profile point ────────────────────────────────
    # Each probe position has several sides. What to do with a side that FAILS
    # the noise test, when forming the per-position amplitude:
    #   "zero"    — count it as 0 absorption and average it in with the passing
    #               sides (a rejected side means "no line resolved here"). Best
    #               for including true beam-edge positions honestly.
    #   "exclude" — drop it; average only the passing sides (position dropped if
    #               none pass).
    #   "keep"    — keep its fitted amplitude anyway, just flag it.
    rejected_mode: str = "zero"
    # A position is FLAGGED (drawn red, "noise-dominated") if fewer than this
    # many sides pass. Under "zero"/"keep" it is still plotted and fitted; under
    # "exclude" a flagged position with zero passing sides is dropped.
    min_good_sides: int = 1
    # Per-position error bar:
    #   "STD"         — standard deviation across the combined sides (incl. any
    #                   zeros) — the side-to-side scatter (thesis choice)
    #   "SEM"         — that divided by sqrt(N)
    #   "upper_bound" — conservative noise-derived bound (snr_min * noise RMS),
    #                   ignoring scatter
    error_mode: str = "STD"

    # ── spatial profile fitting ──────────────────────────────────────────────
    profile_models: list = field(default_factory=lambda: ["gaussian", "lorentzian"])
    profile_offset: bool = False    # allow a constant offset in the profile fit

    # ── geometry (inches, matching the thesis) ───────────────────────────────
    nozzle_diameter_in: float = 0.1     # inscribed-circle diameter of the capillary array
    z_probe_in: float = 2.46            # nozzle -> probe distance
    z_substrate_in: float = 19.685      # nozzle -> growth substrate
    aperture_radius_cm: float = 0.5     # capture-efficiency aperture half-width

    # ── flux measurement ─────────────────────────────────────────────────────
    oven_temp_C: float = 230.0          # nozzle temperature (sets mean atomic speed)
    sigma_eff_cm2: float = 1.246e-9     # resonant cross-section, isotropic pol, 85Rb D2 F=3->F'=4
    f_F3: float = 7.0 / 12.0            # thermal ground-state fraction in F=3 (degeneracy 7/12)
    p_Rb85: float = 0.7217              # 85Rb natural abundance
    # Ar flow in SCCM for the dopant-density estimate; None skips that calc.
    ar_flow_sccm: float | None = None
    ar_density_crystal_kg_m3: float = 1768.0  # solid Ar density for dopant-density calc

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Write this configuration to a JSON file."""
        d = asdict(self)
        # tuples aren't JSON-native; store as lists
        for key in ("match_widths_rel", "gamma_rel_bounds"):
            d[key] = list(d[key])
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        """Load a configuration from a JSON file (unknown keys are ignored)."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in valid}
        for key in ("match_widths_rel", "gamma_rel_bounds"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)
