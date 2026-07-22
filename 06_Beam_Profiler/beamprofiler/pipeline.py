"""
End-to-end pipeline: raw run-format scope CSVs in, beam characterization out.

    results/
    ├── config_used.json          exact configuration of this run
    ├── 01_raw_traces/            trace + detected ramp segmentation, per position
    ├── 02_side_fits/             peak-aware background & Lorentzian fit, per side
    ├── 03_profile/               absorption_profile.csv + per-model profile plots
    ├── summary_<model>.png       final column-density plot + results table
    └── results.json              all numbers, machine-readable

Two passes over the data: the first fits every side freely to find the strongest
position and its linewidth; the second re-fits every side with that linewidth as
the matched-filter template scale (peak positions are still found independently
per side, because the laser drift is not predictable). Sides are then combined
into one amplitude per probe position.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from . import plotting
from .flux import flux_from_absolute_profile
from .peak_fitting import SideResult, analyze_side, reference_gamma_from
from .profile_analysis import ProfileFit, fit_profile
from .trace_io import IN_TO_THOU, Trace, load_trace, read_metadata, segment_all


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionResult:
    position: float                       # stage units [thou]
    side_results: list = field(default_factory=list)   # list[(label, SideResult)]
    amplitude: float = 0.0                # combined over sides
    amplitude_err: float = 0.0
    noise_floor: float = 0.0              # median per-side noise RMS [%]
    n_good_sides: int = 0
    n_sides: int = 0
    flagged: bool = False                 # fewer than min_good_sides passed (drawn red)
    include_in_profile: bool = True
    dc_mean: float | None = None
    n_files: int = 1


def _log(progress, msg: str):
    if progress:
        progress(msg.encode("ascii", errors="replace").decode("ascii"))


def discover_position_files(folder: str | Path) -> dict[float, list[Path]]:
    """Map stage position [thou] -> run-format CSV paths (point_*.csv)."""
    folder = Path(folder)
    out: dict[float, list[Path]] = {}
    for p in sorted(folder.glob("point_*.csv")):
        meta = read_metadata(str(p))
        if "target_in" not in meta:
            continue
        try:
            pos = float(meta["target_in"]) * IN_TO_THOU
        except ValueError:
            continue
        out.setdefault(pos, []).append(p)
    return out


# ──────────────────────────────────────────────────────────────────────────────
def _combine_sides(sides: list[SideResult], config: AnalysisConfig) -> PositionResult:
    """Combine one position's sides into a single amplitude ± error.

    A side that fails the noise test contributes according to `rejected_mode`:
      zero    -> counted as 0 % and averaged in with the passing sides,
      exclude -> dropped (position dropped if none pass),
      keep    -> its fitted amplitude is kept and averaged in anyway.
    """
    pr = PositionResult(position=0.0)
    pr.n_sides = len(sides)
    good = [s for s in sides if s.peak.accepted]
    pr.n_good_sides = len(good)
    noise = [s.sigma for s in sides if np.isfinite(s.sigma)]
    pr.noise_floor = float(np.median(noise)) if noise else 0.0

    mode = config.rejected_mode
    if mode == "exclude":
        values = [s.peak.amplitude for s in good]
    elif mode == "keep":
        values = [s.peak.amplitude for s in sides if s.peak.popt is not None]
    else:  # "zero"
        values = [(s.peak.amplitude if s.peak.accepted else 0.0)
                  for s in sides if s.peak.popt is not None or not s.peak.accepted]

    pr.flagged = pr.n_good_sides < config.min_good_sides

    if not values:
        pr.amplitude = 0.0
        pr.include_in_profile = (mode != "exclude")
        pr.amplitude_err = config.snr_min * max(pr.noise_floor, 1e-9)
        return pr

    values = np.asarray(values, dtype=float)
    n = len(values)
    pr.amplitude = float(np.mean(values))
    pr.include_in_profile = True

    if config.error_mode == "upper_bound":
        err = config.snr_min * pr.noise_floor
    else:
        if n > 1:
            spread = float(np.std(values, ddof=1))
        else:
            spread = float(good[0].peak.amplitude_err) if good else pr.noise_floor
        err = spread / np.sqrt(n) if config.error_mode == "SEM" else spread
    pr.amplitude_err = max(err, pr.noise_floor / np.sqrt(max(n, 1)))
    return pr


def run_extraction(data_folder: str, output_folder: str, config: AnalysisConfig,
                   progress=None, label: str = "") -> tuple[pd.DataFrame, list]:
    """Stages 1-3: extract the absolute spatial absorption profile."""
    out = Path(output_folder)
    dir_raw, dir_fits, dir_prof = out / "01_raw_traces", out / "02_side_fits", out / "03_profile"
    for d in (dir_raw, dir_fits, dir_prof):
        d.mkdir(parents=True, exist_ok=True)

    files = discover_position_files(data_folder)
    if not files:
        raise FileNotFoundError(
            f"No run-format position CSVs (point_*.csv with a '# target_in' header) "
            f"found in {data_folder}")
    n_files = sum(len(v) for v in files.values())
    _log(progress, f"Found {n_files} files at {len(files)} positions in {data_folder}")

    # ── load all traces, then segment the whole run with a shared sweep geometry ─
    traces: dict[float, list[Trace]] = {}
    flat: list[Trace] = []
    for pos, paths in files.items():
        traces[pos] = []
        for path in paths:
            tr = load_trace(str(path), config)
            if tr.dc is None:
                raise ValueError(f"{path.name}: no CH2 DC column — absolute absorption "
                                 "needs the CH2 channel in every file.")
            traces[pos].append(tr)
            flat.append(tr)
    period, duty = segment_all(flat, config)
    _log(progress, f"Segmented {len(flat)} traces (sweep period {period * 1e3:.2f} ms, "
                   f"duty {duty:.2f}).")
    for pos, paths in files.items():
        for path, tr in zip(paths, traces[pos]):
            if len(tr.sides) != config.n_sides_expected:
                _log(progress, f"  note: {path.name}: {len(tr.sides)} ramp sides "
                               f"(expected {config.n_sides_expected})")
            plotting.plot_trace_segmentation(
                tr, str(dir_raw / f"{path.stem}_trace.png"),
                title=f"{label} position {pos:g} — {path.stem} — segmentation")

    def analyze_all(ref_gamma):
        res: dict[float, list[SideResult]] = {}
        for pos, trs in traces.items():
            res[pos] = []
            for tr in trs:
                for side in tr.sides:
                    res[pos].append(analyze_side(
                        side.t, side.v, side.dc_mean, side.dc_err, config, ref_gamma))
        return res

    # ── pass 1: free fits -> reference linewidth ─────────────────────────────
    _log(progress, "Pass 1: locating the strongest position and its linewidth ...")
    first = analyze_all(ref_gamma=None)
    ref_pos, ref_gamma = reference_gamma_from(first)
    if ref_gamma is None:
        raise RuntimeError("No valid absorption peak found at any position — check "
                           "the data, segmentation, and thresholds (snr_min / delta_aic_min).")
    _log(progress, f"Reference position {ref_pos:g}: HWHM = {ref_gamma * 1e6:.1f} us "
                   f"of scan time (matched-filter template scale).")

    # ── pass 2: constrained matched-filter fits ──────────────────────────────
    _log(progress, "Pass 2: fitting every side with the reference template ...")
    final = analyze_all(ref_gamma=ref_gamma)

    # ── combine sides, save plots & tables ───────────────────────────────────
    results: list[PositionResult] = []
    side_rows = []
    for pos in sorted(traces):
        sides_here = final[pos]           # SideResults, in the order analyze_all built them
        # Re-pair each SideResult with its RampSide and a stable label. The order
        # here matches analyze_all exactly (same nested trace/side iteration).
        side_objs, stems = [], []
        for tr in traces[pos]:
            for side in tr.sides:
                side_objs.append(side)
                stems.append(f"{Path(tr.path).stem}:s{side.index}")
        pr = _combine_sides(sides_here, config)
        pr.position = pos
        pr.n_files = len(traces[pos])
        dcs = [s.dc_mean for s in side_objs if s.dc_mean is not None]
        pr.dc_mean = float(np.mean(dcs)) if dcs else None

        for side, sr, stem in zip(side_objs, sides_here, stems):
            pr.side_results.append((stem, sr))
            plotting.plot_side_fit(
                side, sr, str(dir_fits / f"{stem.replace(':', '_')}.png"),
                title=f"{label} position {pos:g} — {stem}")
            side_rows.append({
                "position": pos, "side": stem,
                "amplitude": sr.peak.amplitude, "amplitude_err": sr.peak.amplitude_err,
                "center_ms": sr.peak.center * 1e3, "gamma_us": sr.peak.gamma * 1e6,
                "snr": sr.peak.snr, "delta_aic": sr.peak.delta_aic,
                "accepted": sr.peak.accepted,
                "reject_reasons": "; ".join(sr.peak.reject_reasons),
                "bg_fallback": sr.ramp.used_fallback,
                "dc_mean_V": side.dc_mean,
            })
        results.append(pr)

    for pr in results:
        tag = "ok" if not pr.flagged else "FLAGGED (noise)"
        _log(progress, f"  position {pr.position:g}: {pr.n_good_sides}/{pr.n_sides} sides good"
                       f"  -> A = {pr.amplitude:.4g} +- {pr.amplitude_err:.2g} %  [{tag}]")

    rows = [{
        "position": pr.position, "amplitude": pr.amplitude, "amplitude_err": pr.amplitude_err,
        "n_good_sides": pr.n_good_sides, "n_sides": pr.n_sides, "n_files": pr.n_files,
        "flagged": pr.flagged, "included": pr.include_in_profile,
        "noise_floor": pr.noise_floor, "dc_mean_V": pr.dc_mean,
    } for pr in results]
    df = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    df.to_csv(dir_prof / "absorption_profile.csv", index=False)
    pd.DataFrame(side_rows).to_csv(dir_prof / "side_details.csv", index=False)
    _log(progress, f"Wrote {dir_prof / 'absorption_profile.csv'} (+ side_details.csv)")

    plotting.plot_transmission(
        df, str(dir_prof / "transmission_vs_position.png"),
        title=f"{label} — CH2 DC transmission level vs position")
    return df, results


# ──────────────────────────────────────────────────────────────────────────────
def run_profile(df: pd.DataFrame, output_folder: str, config: AnalysisConfig,
                progress=None, label: str = "") -> dict[str, ProfileFit]:
    """Fit the configured models to the spatial profile; save plots."""
    dir_prof = Path(output_folder) / "03_profile"
    dir_prof.mkdir(parents=True, exist_ok=True)

    x = df["position"].to_numpy(dtype=float)
    y = df["amplitude"].to_numpy(dtype=float)
    e = df["amplitude_err"].to_numpy(dtype=float)
    included = df["included"].to_numpy(dtype=bool)
    accepted = ~df["flagged"].to_numpy(dtype=bool)

    fits: dict[str, ProfileFit] = {}
    for model in config.profile_models:
        pf = fit_profile(x, y, e, model, config, included=included, accepted=accepted)
        fits[model] = pf
        plotting.plot_profile(pf, config, str(dir_prof / f"profile_{model}.png"),
                              dataset_label=f"{label} — {model} profile fit")
        _log(progress, f"  {model:>12}: FWHM = {pf.fwhm_mm:.2f} +- {pf.fwhm_mm_err:.2f} mm, "
                       f"theta_1/2 = {pf.theta_half_deg:.3f} +- {pf.theta_half_deg_err:.3f} deg, "
                       f"chi2_red = {pf.chi2_reduced:.2f}"
                       + (f", eta = {pf.eta:.2f}" if pf.eta is not None else ""))
    return fits


# ──────────────────────────────────────────────────────────────────────────────
def run_full_pipeline(data_folder: str, output_folder: str,
                      config: AnalysisConfig | None = None,
                      progress=None, label: str = "") -> dict:
    """Raw run-format data folder in — final results out."""
    config = config or AnalysisConfig()
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config_used.json")
    label = label or Path(data_folder).name

    _log(progress, f"=== Stage 1-3: extracting absorption profile ({label}) ===")
    df, _ = run_extraction(data_folder, str(out), config, progress, label)

    _log(progress, "=== Stage 4: spatial profile fits ===")
    fits = run_profile(df, str(out), config, progress, label)

    _log(progress, "=== Stage 5: absolute flux (from the absolute profile) ===")
    flux_results = {}
    for model, pf in fits.items():
        fr = flux_from_absolute_profile(pf, config)
        flux_results[model] = fr
        _log(progress, f"  {model:>12}: peak absorption = {fr.absorption_frac * 100:.3f} %, "
                       f"Rb flux = {fr.rb_flux_atoms_s:.2e} atoms/s"
                       + (f", dopant density = {fr.dopant_density_cm3:.2e} cm^-3"
                          if fr.dopant_density_cm3 is not None else ""))

    _log(progress, "=== Writing summaries ===")
    for model, pf in fits.items():
        plotting.plot_summary(pf, flux_results.get(model), config,
                              str(out / f"summary_{model}.png"),
                              dataset_label=f"{label} | {model}")

    results = {"label": label, "data_folder": str(data_folder),
               "absorption_mode": "absolute", "zone1": {}, "flux": {}}
    for model, pf in fits.items():
        results["zone1"][model] = {
            "fwhm_mm": pf.fwhm_mm, "fwhm_mm_err": pf.fwhm_mm_err,
            "theta_half_deg": pf.theta_half_deg, "theta_half_deg_err": pf.theta_half_deg_err,
            "capture_eff": pf.capture_eff, "capture_eff_err": pf.capture_eff_err,
            "chi2_reduced": pf.chi2_reduced, "r_squared": pf.r_squared, "eta": pf.eta,
            "params": dict(zip(pf.param_names, [float(v) for v in pf.popt])),
            "param_errs": dict(zip(pf.param_names, [float(v) for v in pf.perr])),
        }
    for model, fr in flux_results.items():
        results["flux"][model] = {
            "absorption_pct": fr.absorption_frac * 100.0, "optical_depth": fr.optical_depth,
            "n_col_cm2": fr.n_col_cm2, "rb_flux_atoms_s": fr.rb_flux_atoms_s,
            "number_density_cm3": fr.number_density_cm3,
            "reference_position": fr.reference_position,
            "ar_flux_atoms_s": fr.ar_flux_atoms_s, "dopant_density_cm3": fr.dopant_density_cm3,
        }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(progress, f"Done. Results in {out}")
    return results
