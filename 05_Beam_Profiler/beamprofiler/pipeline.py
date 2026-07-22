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
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from . import plotting
from .flux import flux_from_absolute_profile, flux_from_profile
from .peak_fitting import PeakFit, ReferencePeak, fit_side, make_reference
from .profile_analysis import ProfileFit, divergence_method_b, fit_profile
from .ramp_removal import (find_linear_core, subtract_ramp,
                           subtract_ramp_corners, subtract_ramp_wings)
from .trace_io import IN_TO_THOU, Trace, load_trace, read_run_metadata, segment


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SideBundle:
    """Everything produced for one ramp side of one trace."""
    side: object                  # RampSide (full, segmentation-trimmed arrays)
    ramp: object                  # RampFit over the linear core
    peak: PeakFit
    t_used: np.ndarray            # time actually fitted (linear core)
    y_used: np.ndarray            # fitted signal (V, or % absorption)
    sigma_used: float             # noise RMS in the same units as y_used
    core: np.ndarray              # bool mask over side.t (True = kept)
    label: str                    # "<file stem>:s<index>" for reporting
    method: str = "masked"        # background method actually used on this side


@dataclass
class PositionResult:
    """Combined result for one probe position (one or more CSV files)."""
    position: float                      # stage units [thou]
    side_fits: list = field(default_factory=list)   # list[(label, PeakFit)]
    amplitude: float = 0.0               # mean over accepted sides
    amplitude_err: float = 0.0           # STD or SEM over accepted sides
    noise_floor: float = 0.0             # median background noise RMS (fit units)
    n_good_sides: int = 0
    accepted: bool = False
    dc_mean: float | None = None         # mean CH2 transmission level [V] (run format)
    n_files: int = 1


def _log(progress, msg: str):
    if progress:
        # Keep log output ASCII-safe: Windows consoles often use cp1252 and
        # crash on Greek letters / superscripts.
        progress(msg.encode("ascii", errors="replace").decode("ascii"))


def discover_position_files(folder: str | Path) -> dict[float, list[Path]]:
    """Map stage position [thou] -> CSV paths.

    Run-format folders (point_*.csv with a '# target_in' header) take priority:
    the position comes from the metadata (inches -> thou) and several files may
    share one position (multi-pass scans). Otherwise, legacy numerically-named
    files (450.csv = position in thou) are used; non-numeric names (full0.csv,
    RelativeAbsFar.csv, ...) are ignored — those are calibration/derived files.
    """
    folder = Path(folder)
    out: dict[float, list[Path]] = {}
    run_files = [p for p in sorted(folder.glob("point_*.csv"))
                 if "target_in" in read_run_metadata(str(p))]
    if run_files:
        for p in run_files:
            pos = float(read_run_metadata(str(p))["target_in"]) * IN_TO_THOU
            out.setdefault(pos, []).append(p)
        return out
    for p in sorted(folder.glob("*.csv")):
        if re.fullmatch(r"\d+(\.\d+)?", p.stem):
            out.setdefault(float(p.stem), []).append(p)
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
    # Wide enough to cover the Lorentzian wings (4Γ ≈ 6% of peak) and modest
    # centre wander, but no wider — an over-sized mask starves the background
    # fit of anchor points.
    half_window = max(4.0 * reference.gamma,
                      0.5 * config.center_tolerance_frac * duration)
    return (side.t > center - half_window) & (side.t < center + half_window)


def _fit_trace_sides(trace: Trace, config: AnalysisConfig,
                     reference: ReferencePeak | None,
                     percent_mode: bool = False,
                     absolute_mode: bool = False) -> list[SideBundle]:
    """Background-subtract and Lorentzian-fit every side of a trace.

    With `config.auto_trim`, each side is first restricted to its detected
    linear core (the curved ramp ends from the optics cannot be represented by
    the background polynomial and would masquerade as signal).

    percent_mode:  convert the subtracted signal to percent absorption
                   normalized by the FITTED BACKGROUND (legacy unattenuated
                   flux traces, where CH2 carries the full DC level).
    absolute_mode: convert to percent absorption normalized by the CH2 DC
                   level of the side (run format: CH3 dip / CH2 transmission),
                   so every position yields a true absorption percentage.
    """
    stem = Path(trace.path).stem
    # "auto" degree: 1 for run-format (raw) traces — linear once the curved
    # ends are trimmed; a quadratic would absorb part of the broad dip — and
    # 2 for legacy amplified traces (real RC-filter curvature).
    degree = config.ramp_degree
    if not isinstance(degree, int):
        degree = 1 if trace.dc is not None else 2
    # "corners" defines a straight line through the two turn-point clusters, so
    # it only applies when the effective background is linear; curved legacy
    # backgrounds (degree 2) fall back to the sigma-clip method.
    method = config.background_method
    if method == "corners" and degree != 1:
        method = "masked"
    out = []
    for side in trace.sides:
        pre_mask = (_expected_feature_mask(side, reference, config)
                    if reference is not None else None)

        # Bootstrap peak estimate when no reference exists yet (reference-
        # selection pass): both the core detection and the wings background
        # need to know roughly where the peak is. ("corners" needs neither —
        # its anchors are fixed at the side ends.)
        pf0 = None
        if reference is None and method != "corners" and (config.auto_trim or method == "wings"):
            ramp0 = subtract_ramp(
                side.t, side.v, degree=degree,
                clip_sigma=config.clip_sigma, clip_iterations=config.clip_iterations)
            pf0 = fit_side(side.t, ramp0.subtracted, ramp0.sigma_noise, config,
                           reference=None, side_index=side.index)

        n_pts = len(side.t)
        core = np.ones(n_pts, dtype=bool)
        if method == "corners":
            # The background line is anchored on the corner clusters of the
            # FULL side; the peak fit then runs on the interior only, so the
            # curved ends under the anchors cannot inflate the linewidth.
            a = int(config.corner_fit_exclude_fraction * n_pts)
            if a > 0:
                core[:a] = False
                core[n_pts - a:] = False
        elif config.auto_trim:
            trim_mask = pre_mask
            if trim_mask is None and pf0 is not None and pf0.popt is not None:
                duration = float(side.t[-1] - side.t[0])
                half_w = min(4.0 * max(pf0.gamma, 1e-9), 0.30 * duration)
                trim_mask = (side.t > pf0.center - half_w) & (side.t < pf0.center + half_w)
            core = find_linear_core(
                side.t, side.v, degree=degree,
                clip_sigma=config.clip_sigma, clip_iterations=config.clip_iterations,
                known_feature_mask=trim_mask,
                max_trim_fraction=config.auto_trim_max_fraction,
                resid_sigma=config.auto_trim_sigma,
            )
        tc, vc = side.t[core], side.v[core]

        # Where should the background expect the peak? (same convention as
        # fit_side: reference centre fraction applies to the fitted window)
        center_est = gamma_est = None
        if reference is not None:
            frac = reference.center_frac.get(
                side.index, float(np.mean(list(reference.center_frac.values()))))
            center_est = float(tc[0] + frac * (tc[-1] - tc[0]))
            gamma_est = reference.gamma
        elif pf0 is not None and pf0.popt is not None:
            center_est, gamma_est = pf0.center, max(pf0.gamma, 1e-9)

        if method == "corners":
            # Anchor the line on the FULL side (the clusters sit at the ends);
            # only the interior enters the peak fit.
            ramp = subtract_ramp_corners(
                side.t, side.v, cluster_fraction=config.corner_cluster_fraction)
            sub_fit = ramp.subtracted[core]
        elif method == "wings" and center_est is not None:
            ramp = subtract_ramp_wings(
                tc, vc, center=center_est,
                inner_halfwidth=config.wing_inner_gammas * gamma_est,
                degree=degree)
            sub_fit = ramp.subtracted
        else:
            ramp = subtract_ramp(
                tc, vc, degree=degree,
                clip_sigma=config.clip_sigma, clip_iterations=config.clip_iterations,
                known_feature_mask=(pre_mask[core] if pre_mask is not None else None),
            )
            sub_fit = ramp.subtracted
        if absolute_mode and side.dc_mean is not None:
            # True absorption percent: CH3 dip depth over CH2 DC transmission.
            denom = side.dc_mean - config.dark_offset_V
            y = 100.0 * sub_fit / denom
            sigma = 100.0 * ramp.sigma_noise / denom
        elif percent_mode:
            # Guard against non-physical backgrounds near zero volts.
            bg = np.where(np.abs(ramp.background) > 1e-6, ramp.background, np.nan)
            y = 100.0 * ramp.subtracted / bg
            y = np.nan_to_num(y, nan=0.0)
            sigma = 100.0 * ramp.sigma_noise / float(np.nanmedian(np.abs(bg)))
        else:
            y, sigma = sub_fit, ramp.sigma_noise

        pf = fit_side(tc, y, sigma, config,
                      reference=reference, side_index=side.index)
        if absolute_mode and side.dc_mean is not None and pf.popt is not None:
            # Fold the DC-normalization uncertainty (scatter + ADC quantization
            # of CH2) into the amplitude error.
            rel_dc = (side.dc_err or 0.0) / max(side.dc_mean - config.dark_offset_V, 1e-9)
            pf.amplitude_err = float(np.hypot(pf.amplitude_err, pf.amplitude * rel_dc))
        out.append(SideBundle(side=side, ramp=ramp, peak=pf, t_used=tc,
                              y_used=y, sigma_used=sigma, core=core,
                              label=f"{stem}:s{side.index}", method=method))
    return out


def run_extraction(data_folder: str, output_folder: str, config: AnalysisConfig,
                   progress=None, label: str = "") -> tuple[pd.DataFrame, list, bool]:
    """Stages 1-3 for every position CSV in `data_folder`.

    Returns (profile DataFrame, list[PositionResult], absolute_mode) and writes
    absorption_profile.csv (+ side_details.csv) plus all stage images under
    `output_folder`. In absolute mode the amplitudes are true percent
    absorption (CH3 dip normalized by the CH2 DC level per side).
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
            f"No position CSVs (point_*.csv with a target_in header, or numeric "
            f"filenames like 450.csv) found in {data_folder}")
    n_files = sum(len(v) for v in files.values())
    _log(progress, f"Found {n_files} files at {len(files)} positions in {data_folder}")

    # ── load & segment all traces (kept in memory: sides are downsampled) ────
    traces: dict[float, list[Trace]] = {}
    for pos, paths in files.items():
        traces[pos] = []
        for path in paths:
            tr = segment(load_trace(str(path), config), config)
            if len(tr.sides) != config.n_sides_expected:
                _log(progress, f"  note: {path.name}: {len(tr.sides)} ramp sides "
                               f"(expected {config.n_sides_expected})")
            traces[pos].append(tr)
            plotting.plot_trace_segmentation(
                tr, str(dir_raw / f"{path.stem}_trace.png"),
                title=f"{label} position {pos:g} — {path.stem} — segmentation")
    _log(progress, "All traces loaded and segmented.")

    # ── absorption scale: absolute (CH3/CH2, run format) or relative ─────────
    have_dc = all(tr.dc is not None for trs in traces.values() for tr in trs)
    if config.absorption_mode == "absolute" and not have_dc:
        raise ValueError("absorption_mode='absolute' requires the run data format "
                         "with a CH2 DC column in every file.")
    absolute_mode = (config.absorption_mode == "absolute"
                     or (config.absorption_mode == "auto" and have_dc))
    unit = "% absorption" if absolute_mode else "V (relative)"
    _log(progress, f"Absorption scale: {'ABSOLUTE (CH3 dip / CH2 DC level)' if absolute_mode else 'relative'}"
                   + (f", dark offset {config.dark_offset_V:g} V" if absolute_mode and config.dark_offset_V else ""))

    # ── pick the reference trace: strongest unconstrained peak SNR ───────────
    _log(progress, "Selecting reference (strongest) position ...")
    best_pos, best_snr, best_bundles = None, -np.inf, None
    for pos, trs in traces.items():
        for tr in trs:
            bundles = _fit_trace_sides(tr, config, reference=None,
                                       absolute_mode=absolute_mode)
            snrs = [b.peak.snr for b in bundles if b.peak.popt is not None]
            score = float(np.median(snrs)) if snrs else -np.inf
            if score > best_snr:
                best_pos, best_snr, best_bundles = pos, score, bundles
    _log(progress, f"Reference position: {best_pos:g} (median SNR {best_snr:.1f})")

    # The reference template's centre fractions must refer to the same time
    # windows the fits actually used (the linear cores), so pass those.
    reference = make_reference(
        {b.side.index: b.peak for b in best_bundles},
        [SimpleNamespace(index=b.side.index, t=b.t_used) for b in best_bundles])
    if reference is None:
        raise RuntimeError(
            "No valid absorption peak found at any position — check the data, "
            "segmentation settings, and thresholds (snr_min / delta_aic_min).")
    _log(progress, f"Reference linewidth Gamma(HWHM) = {reference.gamma * 1e6:.1f} us "
                   f"of scan time; centres locked per side.")

    # ── constrained fits for every position ──────────────────────────────────
    results: list[PositionResult] = []
    side_rows = []
    for pos in sorted(traces):
        pr = PositionResult(position=pos, n_files=len(traces[pos]))
        bundles: list[SideBundle] = []
        for tr in traces[pos]:
            bundles += _fit_trace_sides(tr, config, reference=reference,
                                        absolute_mode=absolute_mode)

        for b in bundles:
            pr.side_fits.append((b.label, b.peak))
            plotting.plot_side_fit(
                b.side, b.ramp, b.peak,
                str(dir_fits / f"{b.label.replace(':', '_')}.png"),
                title=f"{label} position {pos:g} — {b.label}",
                y_label="Absorption [%]" if absolute_mode else "Relative absorption [arb. u.]",
                y_fitted=b.y_used, t_fitted=b.t_used, core=b.core,
                method=b.method)
            side_rows.append({
                "position": pos, "side": b.label,
                "amplitude": b.peak.amplitude, "amplitude_err": b.peak.amplitude_err,
                "snr": b.peak.snr, "delta_aic": b.peak.delta_aic,
                "accepted": b.peak.accepted,
                "reject_reasons": "; ".join(b.peak.reject_reasons),
                "core_fraction": float(np.mean(b.core)),
                "dc_mean_V": b.side.dc_mean,
            })

        good = [pf for (_, pf) in pr.side_fits if pf.accepted]
        noise = [b.sigma_used for b in bundles if np.isfinite(b.sigma_used)]
        pr.noise_floor = float(np.median(noise)) if noise else 0.0
        pr.n_good_sides = len(good)
        pr.accepted = len(good) >= config.min_good_sides
        dcs = [b.side.dc_mean for b in bundles if b.side.dc_mean is not None]
        pr.dc_mean = float(np.mean(dcs)) if dcs else None

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
                       f"-> A = {pr.amplitude:.4g} +- {pr.amplitude_err:.2g} {unit}  [{tag}]")

    # ── profile table ─────────────────────────────────────────────────────────
    rows = []
    for pr in results:
        rows.append({
            "position": pr.position,
            "amplitude": pr.amplitude,
            "amplitude_err": pr.amplitude_err,
            "n_good_sides": pr.n_good_sides,
            "n_sides": len(pr.side_fits),
            "n_files": pr.n_files,
            "accepted": pr.accepted,
            "noise_floor": pr.noise_floor,
            "dc_mean_V": pr.dc_mean,
        })
    df = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    df.to_csv(dir_prof / "absorption_profile.csv", index=False)
    pd.DataFrame(side_rows).to_csv(dir_prof / "side_details.csv", index=False)
    _log(progress, f"Wrote {dir_prof / 'absorption_profile.csv'} (+ side_details.csv)")

    # Transmission diagnostic: the per-point CH2 DC level tracks window
    # transmission differences across the scan (run format only).
    if absolute_mode:
        plotting.plot_transmission(
            df, str(dir_prof / "transmission_vs_position.png"),
            title=f"{label} — CH2 DC transmission level vs position")
    return df, results, absolute_mode


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: spatial-profile fits
# ──────────────────────────────────────────────────────────────────────────────
def run_profile(df: pd.DataFrame, output_folder: str, config: AnalysisConfig,
                progress=None, label: str = "",
                absolute_mode: bool = False) -> dict[str, ProfileFit]:
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

    y_label = "Absorption [%]" if absolute_mode else "Relative absorption [arb. u.]"
    fits: dict[str, ProfileFit] = {}
    for model in config.profile_models:
        pf = fit_profile(x, y, e, model, config, included=included, accepted=ok)
        fits[model] = pf
        plotting.plot_profile(pf, config,
                              str(dir_prof / f"profile_{model}.png"),
                              dataset_label=f"{label} — {model} profile fit",
                              y_label=y_label)
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
        bundles = _fit_trace_sides(tr, config, reference=None, percent_mode=True)
        for b in bundles:
            plotting.plot_side_fit(
                b.side, b.ramp, b.peak,
                str(dir_flux / f"{path.stem}_side{b.side.index}.png"),
                title=f"flux {path.stem} — side {b.side.index}",
                y_label="Absorption [%]", y_fitted=b.y_used,
                t_fitted=b.t_used, core=b.core, method=b.method)
            if b.peak.accepted:
                amps.append(b.peak.amplitude)
        n_acc = sum(1 for b in bundles if b.peak.accepted)
        _log(progress, f"  flux trace {path.name}: {n_acc}/{len(bundles)} sides accepted")

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
    df, _position_results, absolute_mode = run_extraction(
        data_folder, str(out), config, progress, label)

    _log(progress, "=== Stage 4: spatial profile fits ===")
    fits = run_profile(df, str(out), config, progress, label,
                       absolute_mode=absolute_mode)

    # ── optional Zone 2 for Method B ─────────────────────────────────────────
    fits2, method_b = {}, {}
    if zone2_folder:
        _log(progress, "=== Zone 2: extracting second-zone profile (Method B) ===")
        out2 = out / "zone2"
        cfg2 = AnalysisConfig(**{**config.__dict__})
        cfg2.z_probe_in = config.z_probe2_in    # geometry of the far zone
        df2, _, abs2 = run_extraction(zone2_folder, str(out2), cfg2, progress,
                                      label=f"{label}-zone2")
        fits2 = run_profile(df2, str(out2), cfg2, progress, f"{label}-zone2",
                            absolute_mode=abs2)
        for model in fits:
            if model in fits2:
                th, th_err = divergence_method_b(
                    fits[model].fwhm_mm, fits[model].fwhm_mm_err,
                    fits2[model].fwhm_mm, fits2[model].fwhm_mm_err, config)
                method_b[model] = {"theta_half_deg": th, "theta_half_deg_err": th_err}
                _log(progress, f"  Method B ({model}): theta_1/2 = {th:.3f} +- {th_err:.3f} deg")

    # ── absolute flux ─────────────────────────────────────────────────────────
    # Absolute-mode runs carry true percent absorption at every position, so
    # the flux comes straight from the fitted profile — no calibration trace
    # needed. A flux trace, if explicitly given, takes precedence (legacy path).
    flux_results = {}
    flux_source_used = None
    if flux_source:
        _log(progress, "=== Stage 5: absolute flux (from calibration trace) ===")
        flux_source_used = "trace"
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
    elif absolute_mode:
        _log(progress, "=== Stage 5: absolute flux (from absolute profile) ===")
        flux_source_used = "profile"
        for model, pf in fits.items():
            fr = flux_from_absolute_profile(pf, config)
            flux_results[model] = fr
            _log(progress, f"  {model:>12}: peak absorption = {fr.absorption_frac * 100:.3f} %, "
                           f"Rb flux = {fr.rb_flux_atoms_s:.2e} atoms/s"
                           + (f", dopant density = {fr.dopant_density_cm3:.2e} cm^-3"
                              if fr.dopant_density_cm3 is not None else ""))

    # ── summary figures & results.json ───────────────────────────────────────
    _log(progress, "=== Writing summaries ===")
    for model, pf in fits.items():
        plotting.plot_summary(pf, flux_results.get(model), config,
                              str(out / f"summary_{model}.png"),
                              dataset_label=f"{label} | {model}")

    results = {"label": label, "data_folder": str(data_folder),
               "absorption_mode": "absolute" if absolute_mode else "relative",
               "flux_source": flux_source_used,
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
