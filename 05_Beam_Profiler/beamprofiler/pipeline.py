"""
End-to-end pipeline: raw scope CSVs in, final beam characterization out.

    results/
    ├── config_used.json          exact configuration of this run
    ├── 01_raw_traces/            trace + detected ramp segmentation, per position
    ├── 02_side_fits/             background & Lorentzian fit, per position & side
    ├── 03_profile/               absorption_profile.csv + per-model profile plots
    ├── 04_flux/                  absolute-absorption fits (if flux data given)
    ├── summary_<model>.png       final column-density plot + results table
    └── results.json              all numbers, machine-readable

The stages can also be called individually (see run_extraction, run_profile,
run_flux_extraction) — the GUI and CLI both drive `run_full_pipeline`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from . import plotting
from .flux import flux_from_profile
from .peak_fitting import PeakFit, ReferencePeak, fit_side, make_reference
from .profile_analysis import ProfileFit, divergence_method_b, fit_profile
from .ramp_removal import subtract_ramp
from .trace_io import Trace, load_trace, segment


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionResult:
    """Combined result for one probe position (one CSV file)."""
    position: float                      # stage units (from the filename)
    side_fits: dict = field(default_factory=dict)   # side index -> PeakFit
    amplitude: float = 0.0               # mean over accepted sides
    amplitude_err: float = 0.0           # STD or SEM over accepted sides
    noise_floor: float = 0.0             # median background noise RMS
    n_good_sides: int = 0
    accepted: bool = False


def _log(progress, msg: str):
    if progress:
        # Keep log output ASCII-safe: Windows consoles often use cp1252 and
        # crash on Greek letters / superscripts.
        progress(msg.encode("ascii", errors="replace").decode("ascii"))


def discover_position_files(folder: str | Path) -> dict[float, Path]:
    """Map stage position -> CSV path for numerically-named files (e.g. 450.csv).

    Non-numeric names (full0.csv, RelativeAbsFar.csv, ...) are ignored — those
    are calibration/derived files, not position scans.
    """
    out = {}
    for p in sorted(Path(folder).glob("*.csv")):
        if re.fullmatch(r"\d+(\.\d+)?", p.stem):
            out[float(p.stem)] = p
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1-3: extraction of the spatial absorption profile
# ──────────────────────────────────────────────────────────────────────────────
def _expected_feature_mask(side, reference: ReferencePeak, config: AnalysisConfig):
    """Pre-mask the region where the reference says the peak must be.

    Used during background refitting on weak positions: even if the dip is
    below the sigma-clip threshold, it is excluded from the polynomial fit so
    it cannot bias the background.
    """
    duration = side.t[-1] - side.t[0]
    if side.index in reference.center_frac:
        frac = reference.center_frac[side.index]
    else:
        frac = float(np.mean(list(reference.center_frac.values())))
    center = side.t[0] + frac * duration
    half_window = max(3.0 * reference.gamma,
                      config.center_tolerance_frac * duration)
    return (side.t > center - half_window) & (side.t < center + half_window)


def _fit_trace_sides(trace: Trace, config: AnalysisConfig,
                     reference: ReferencePeak | None,
                     percent_mode: bool = False):
    """Background-subtract and Lorentzian-fit every side of a trace.

    percent_mode: convert the subtracted signal to percent absorption
    (100·ΔV/V_background) — used for the unattenuated flux traces where the
    absolute scale matters.
    Returns list of (side, ramp_fit, peak_fit, y_used, sigma_used).
    """
    out = []
    for side in trace.sides:
        pre_mask = (_expected_feature_mask(side, reference, config)
                    if reference is not None else None)
        ramp = subtract_ramp(
            side.t, side.v, degree=config.ramp_degree,
            clip_sigma=config.clip_sigma, clip_iterations=config.clip_iterations,
            known_feature_mask=pre_mask,
        )
        if percent_mode:
            # Guard against non-physical backgrounds near zero volts.
            bg = np.where(np.abs(ramp.background) > 1e-6, ramp.background, np.nan)
            y = 100.0 * ramp.subtracted / bg
            y = np.nan_to_num(y, nan=0.0)
            sigma = 100.0 * ramp.sigma_noise / float(np.nanmedian(np.abs(bg)))
        else:
            y, sigma = ramp.subtracted, ramp.sigma_noise

        pf = fit_side(side.t, y, sigma, config,
                      reference=reference, side_index=side.index)
        out.append((side, ramp, pf, y, sigma))
    return out


def run_extraction(data_folder: str, output_folder: str, config: AnalysisConfig,
                   progress=None, label: str = "") -> tuple[pd.DataFrame, list]:
    """Stages 1-3 for every position CSV in `data_folder`.

    Returns (profile DataFrame, list[PositionResult]) and writes
    absorption_profile.csv plus all stage images under `output_folder`.
    """
    out = Path(output_folder)
    dir_raw = out / "01_raw_traces"
    dir_fits = out / "02_side_fits"
    dir_prof = out / "03_profile"
    for d in (dir_raw, dir_fits, dir_prof):
        d.mkdir(parents=True, exist_ok=True)

    files = discover_position_files(data_folder)
    if not files:
        raise FileNotFoundError(
            f"No position CSVs (numeric filenames like 450.csv) found in {data_folder}")
    _log(progress, f"Found {len(files)} position files in {data_folder}")

    # ── load & segment all traces (kept in memory: sides are downsampled) ────
    traces: dict[float, Trace] = {}
    for pos, path in files.items():
        tr = segment(load_trace(str(path), config), config)
        if len(tr.sides) != config.n_sides_expected:
            _log(progress, f"  note: {path.name}: {len(tr.sides)} ramp sides "
                           f"(expected {config.n_sides_expected})")
        traces[pos] = tr
        plotting.plot_trace_segmentation(
            tr, str(dir_raw / f"{path.stem}_trace.png"),
            title=f"{label} position {path.stem} — segmentation")
    _log(progress, "All traces loaded and segmented.")

    # ── pick the reference position: strongest unconstrained peak SNR ────────
    _log(progress, "Selecting reference (strongest) position ...")
    best_pos, best_snr, best_fits = None, -np.inf, None
    for pos, tr in traces.items():
        fits = _fit_trace_sides(tr, config, reference=None)
        snrs = [pf.snr for (_, _, pf, _, _) in fits if pf.popt is not None]
        score = float(np.median(snrs)) if snrs else -np.inf
        if score > best_snr:
            best_pos, best_snr, best_fits = pos, score, fits
    _log(progress, f"Reference position: {best_pos:g} (median SNR {best_snr:.1f})")

    reference = make_reference(
        {s.index: pf for (s, _, pf, _, _) in best_fits}, traces[best_pos].sides)
    if reference is None:
        raise RuntimeError(
            "No valid absorption peak found at any position — check the data, "
            "segmentation settings, and thresholds (snr_min / delta_aic_min).")
    _log(progress, f"Reference linewidth Gamma(HWHM) = {reference.gamma * 1e6:.1f} us "
                   f"of scan time; centres locked per side.")

    # ── constrained fits for every position ──────────────────────────────────
    results: list[PositionResult] = []
    for pos in sorted(traces):
        tr = traces[pos]
        fits = _fit_trace_sides(tr, config, reference=reference)
        pr = PositionResult(position=pos)
        stem = files[pos].stem

        for side, ramp, pf, y, sigma in fits:
            pr.side_fits[side.index] = pf
            plotting.plot_side_fit(
                side, ramp, pf,
                str(dir_fits / f"{stem}_side{side.index}.png"),
                title=f"{label} position {stem} — side {side.index}")

        good = [pf for pf in pr.side_fits.values() if pf.accepted]
        noise = [ramp.sigma_noise for (_, ramp, _, _, _) in fits
                 if np.isfinite(ramp.sigma_noise)]
        pr.noise_floor = float(np.median(noise)) if noise else 0.0
        pr.n_good_sides = len(good)
        pr.accepted = len(good) >= config.min_good_sides

        if pr.accepted:
            amps = np.array([pf.amplitude for pf in good])
            pr.amplitude = float(np.mean(amps))
            spread = float(np.std(amps, ddof=1)) if len(amps) > 1 else float(good[0].amplitude_err)
            pr.amplitude_err = (spread / np.sqrt(len(amps))
                                if config.error_mode == "SEM" else spread)
            # never report an error below the fit/noise floor
            pr.amplitude_err = max(pr.amplitude_err, pr.noise_floor / max(np.sqrt(len(amps)), 1.0))
        else:
            # rejected: amplitude 0 with a noise-derived upper bound
            # (final error assigned after the loop, once the global noise
            # floor is known — a corrupt trace must not report sigma ~ 0)
            pr.amplitude = 0.0
            pr.amplitude_err = np.nan
        results.append(pr)
    # Upper-bound errors for rejected positions, floored at the run-wide median
    # noise so a corrupt trace (noise ~ 0) cannot anchor the profile fit.
    median_noise = float(np.median([p.noise_floor for p in results if p.noise_floor > 0]))
    for pr in results:
        if not pr.accepted:
            pr.amplitude_err = config.snr_min * max(pr.noise_floor, median_noise)

    for pr in results:
        tag = "ok" if pr.accepted else "REJECTED (noise)"
        _log(progress, f"  position {pr.position:g}: {pr.n_good_sides}/"
                       f"{len(pr.side_fits)} sides good "
                       f"-> A = {pr.amplitude:.4g} +- {pr.amplitude_err:.2g}  [{tag}]")

    # ── profile table ─────────────────────────────────────────────────────────
    rows = []
    for pr in results:
        row = {
            "position": pr.position,
            "amplitude": pr.amplitude,
            "amplitude_err": pr.amplitude_err,
            "n_good_sides": pr.n_good_sides,
            "accepted": pr.accepted,
            "noise_floor": pr.noise_floor,
        }
        for i, pf in sorted(pr.side_fits.items()):
            row[f"amp_side{i}"] = pf.amplitude
            row[f"amp_err_side{i}"] = pf.amplitude_err
            row[f"snr_side{i}"] = pf.snr
            row[f"accepted_side{i}"] = pf.accepted
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    df.to_csv(dir_prof / "absorption_profile.csv", index=False)
    _log(progress, f"Wrote {dir_prof / 'absorption_profile.csv'}")
    return df, results


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: spatial-profile fits
# ──────────────────────────────────────────────────────────────────────────────
def run_profile(df: pd.DataFrame, output_folder: str, config: AnalysisConfig,
                progress=None, label: str = "") -> dict[str, ProfileFit]:
    """Fit the configured models to the spatial profile; save plots."""
    dir_prof = Path(output_folder) / "03_profile"
    dir_prof.mkdir(parents=True, exist_ok=True)

    x = df["position"].to_numpy(dtype=float)
    y = df["amplitude"].to_numpy(dtype=float)
    e = df["amplitude_err"].to_numpy(dtype=float)
    ok = df["accepted"].to_numpy(dtype=bool)

    # rejected-point policy (thesis Sec. 5.3 — don't let noise inflate the tails)
    if config.rejected_mode == "exclude":
        included = ok.copy()
    else:                       # "zero" (already zeroed upstream) or "keep"
        included = np.ones(len(x), dtype=bool)

    fits: dict[str, ProfileFit] = {}
    for model in config.profile_models:
        pf = fit_profile(x, y, e, model, config, included=included, accepted=ok)
        fits[model] = pf
        plotting.plot_profile(pf, config,
                              str(dir_prof / f"profile_{model}.png"),
                              dataset_label=f"{label} — {model} profile fit")
        _log(progress, f"  {model:>12}: FWHM = {pf.fwhm_mm:.2f} +- {pf.fwhm_mm_err:.2f} mm, "
                       f"theta_1/2 = {pf.theta_half_deg:.3f} +- {pf.theta_half_deg_err:.3f} deg, "
                       f"chi2_red = {pf.chi2_reduced:.2f}"
                       + (f", eta = {pf.eta:.2f}" if pf.eta is not None else ""))
    return fits


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: absolute flux
# ──────────────────────────────────────────────────────────────────────────────
def run_flux_extraction(flux_source: str, output_folder: str, config: AnalysisConfig,
                        progress=None) -> tuple[float, float]:
    """Fit the unattenuated trace(s) and return (absorption fraction, reference position).

    `flux_source` may be a single CSV or a folder of CSVs (all are averaged).
    The reference stage position is config.flux_reference_position, or parsed
    as the first number in each filename.
    """
    dir_flux = Path(output_folder) / "04_flux"
    dir_flux.mkdir(parents=True, exist_ok=True)

    src = Path(flux_source)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"No flux CSVs found at {flux_source}")

    amps, ref_pos = [], config.flux_reference_position
    for path in files:
        if ref_pos is None:
            m = re.search(r"\d+(\.\d+)?", path.stem)
            if m:
                ref_pos = float(m.group(0))
        tr = segment(load_trace(str(path), config), config)
        plotting.plot_trace_segmentation(
            tr, str(dir_flux / f"{path.stem}_trace.png"),
            title=f"flux trace {path.name} — segmentation")
        # Unconstrained fits: the flux trace can have different scan settings
        # than the amplified data, so the reference template is not reused.
        fits = _fit_trace_sides(tr, config, reference=None, percent_mode=True)
        for side, ramp, pf, y, sigma in fits:
            plotting.plot_side_fit(
                side, ramp, pf,
                str(dir_flux / f"{path.stem}_side{side.index}.png"),
                title=f"flux {path.stem} — side {side.index}",
                y_label="Absorption [%]", y_fitted=y)
            if pf.accepted:
                amps.append(pf.amplitude)
        n_acc = sum(1 for (_, _, pf, _, _) in fits if pf.accepted)
        _log(progress, f"  flux trace {path.name}: {n_acc}/{len(fits)} sides accepted")

    if not amps:
        raise RuntimeError("No flux-trace side passed validation — cannot compute flux.")
    absorption_pct = float(np.mean(amps))
    _log(progress, f"Mean absolute absorption: {absorption_pct:.3f} % "
                   f"({len(amps)} accepted sides), reference position {ref_pos}")
    return absorption_pct / 100.0, ref_pos


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_full_pipeline(
    data_folder: str,
    output_folder: str,
    config: AnalysisConfig | None = None,
    flux_source: str | None = None,
    zone2_folder: str | None = None,
    progress=None,
    label: str = "",
) -> dict:
    """Raw data folder in — final results out.

    Parameters
    ----------
    data_folder   : folder of position CSVs (Zone 1)
    output_folder : where all results/images are written
    config        : AnalysisConfig (defaults used if None)
    flux_source   : optional unattenuated trace file/folder for absolute flux
    zone2_folder  : optional second-zone position CSVs for Method B divergence
    progress      : optional callable(str) for live logging (GUI/CLI)
    """
    config = config or AnalysisConfig()
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config_used.json")
    label = label or Path(data_folder).name

    _log(progress, f"=== Stage 1-3: extracting absorption profile ({label}) ===")
    df, _position_results = run_extraction(data_folder, str(out), config, progress, label)

    _log(progress, "=== Stage 4: spatial profile fits ===")
    fits = run_profile(df, str(out), config, progress, label)

    # ── optional Zone 2 for Method B ─────────────────────────────────────────
    fits2, method_b = {}, {}
    if zone2_folder:
        _log(progress, "=== Zone 2: extracting second-zone profile (Method B) ===")
        out2 = out / "zone2"
        cfg2 = AnalysisConfig(**{**config.__dict__})
        cfg2.z_probe_in = config.z_probe2_in    # geometry of the far zone
        df2, _ = run_extraction(zone2_folder, str(out2), cfg2, progress,
                                label=f"{label}-zone2")
        fits2 = run_profile(df2, str(out2), cfg2, progress, f"{label}-zone2")
        for model in fits:
            if model in fits2:
                th, th_err = divergence_method_b(
                    fits[model].fwhm_mm, fits[model].fwhm_mm_err,
                    fits2[model].fwhm_mm, fits2[model].fwhm_mm_err, config)
                method_b[model] = {"theta_half_deg": th, "theta_half_deg_err": th_err}
                _log(progress, f"  Method B ({model}): theta_1/2 = {th:.3f} +- {th_err:.3f} deg")

    # ── optional absolute flux ────────────────────────────────────────────────
    flux_results = {}
    if flux_source:
        _log(progress, "=== Stage 5: absolute flux ===")
        absorption_frac, ref_pos = run_flux_extraction(flux_source, str(out), config, progress)
        if ref_pos is None:
            ref_pos = fits[list(fits)[0]].center
            _log(progress, f"  no reference position given/parsed — using profile centre {ref_pos:.0f}")
        for model, pf in fits.items():
            fr = flux_from_profile(absorption_frac, pf, config, ref_pos)
            flux_results[model] = fr
            _log(progress, f"  {model:>12}: Rb flux = {fr.rb_flux_atoms_s:.2e} atoms/s"
                           + (f", dopant density = {fr.dopant_density_cm3:.2e} cm^-3"
                              if fr.dopant_density_cm3 is not None else ""))

    # ── summary figures & results.json ───────────────────────────────────────
    _log(progress, "=== Writing summaries ===")
    for model, pf in fits.items():
        plotting.plot_summary(pf, flux_results.get(model), config,
                              str(out / f"summary_{model}.png"),
                              dataset_label=f"{label} | {model}")

    results = {"label": label, "data_folder": str(data_folder),
               "zone1": {}, "zone2": {}, "method_b": method_b, "flux": {}}
    for model, pf in fits.items():
        results["zone1"][model] = {
            "fwhm_mm": pf.fwhm_mm, "fwhm_mm_err": pf.fwhm_mm_err,
            "theta_half_deg": pf.theta_half_deg,
            "theta_half_deg_err": pf.theta_half_deg_err,
            "capture_eff": pf.capture_eff, "capture_eff_err": pf.capture_eff_err,
            "chi2_reduced": pf.chi2_reduced, "r_squared": pf.r_squared,
            "eta": pf.eta,
            "params": dict(zip(pf.param_names, [float(v) for v in pf.popt])),
            "param_errs": dict(zip(pf.param_names, [float(v) for v in pf.perr])),
        }
    for model, pf in fits2.items():
        results["zone2"][model] = {
            "fwhm_mm": pf.fwhm_mm, "fwhm_mm_err": pf.fwhm_mm_err,
            "theta_half_deg": pf.theta_half_deg,
            "theta_half_deg_err": pf.theta_half_deg_err,
            "chi2_reduced": pf.chi2_reduced,
        }
    for model, fr in flux_results.items():
        results["flux"][model] = {
            "absorption_pct": fr.absorption_frac * 100.0,
            "optical_depth": fr.optical_depth,
            "n_col_cm2": fr.n_col_cm2,
            "rb_flux_atoms_s": fr.rb_flux_atoms_s,
            "number_density_cm3": fr.number_density_cm3,
            "reference_position": fr.reference_position,
            "ar_flux_atoms_s": fr.ar_flux_atoms_s,
            "dopant_density_cm3": fr.dopant_density_cm3,
        }

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(progress, f"Done. Results in {out}")
    return results
