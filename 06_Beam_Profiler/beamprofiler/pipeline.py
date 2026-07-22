"""
End-to-end pipeline: raw run-format scope CSVs in, beam characterization out.

    results/
    ├── config_used.json          exact configuration of this run
    ├── manual_overrides.json     hand-set per-side verdicts, if any (see side_overrides)
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

Manual verdicts from the GUI (`manual_overrides.json`) are applied between those
two steps — after the final fits, before the sides are combined — so a side kept
or dropped by hand changes the profile but never the reference linewidth.
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
from .side_overrides import apply_override, load_overrides
from .trace_io import IN_TO_THOU, Trace, load_trace, read_metadata, segment_all


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionResult:
    position: float                       # stage units [thou]
    side_results: list = field(default_factory=list)   # list[(label, SideResult)]
    amplitude: float = 0.0                # combined over sides
    amplitude_err: float = 0.0
    noise_floor: float = 0.0              # median per-side noise RMS [%]
    ramp_span_V: float = 0.0              # mean CH3 sawtooth span over the fitted sides [V]
    ramp_span_err_V: float = 0.0          # its side-to-side scatter [V]
    n_good_sides: int = 0
    n_sides: int = 0
    n_overridden: int = 0                 # sides whose verdict was set by hand
    n_suspect_sides: int = 0              # accepted-but-flagged sides (flag mode)
    flagged: bool = False                 # fewer than min_good_sides passed (drawn red)
    include_in_profile: bool = True
    dc_mean: float | None = None
    n_files: int = 1


def side_image_name(stem: str) -> str:
    """PNG filename in 02_side_fits for a side label like 'point_0004:s3'."""
    return stem.replace(":", "_") + ".png"


def ramp_span_volts(side_result: SideResult) -> float:
    """Peak-to-peak span of the laser-power sawtooth over one fitted side [V].

    The background under a side is the straight sweep ramp, so its end-to-end
    span IS the triangle wave's trough-to-crest amplitude — measured on CH3, the
    channel the fits run on. It is reported as measured, i.e. over the side that
    was actually fitted (edge-trimmed by `edge_trim_fraction` at each end, and
    with the peak window excluded from the anchor). It is deliberately NOT
    extrapolated back to the untrimmed side: the trimmed-off part is the ramp
    apex, which is exactly where the sweep is curved rather than linear.
    """
    bg = side_result.ramp.background
    if bg is None or len(bg) < 2:
        return float("nan")
    return abs(float(bg[-1] - bg[0]))


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

    A hand-set verdict (see side_overrides) has already been folded into
    `peak.accepted` by this point, so a forced reject follows exactly the same
    `rejected_mode` route as an automatic one — under `keep` it therefore only
    changes the good-side count, not the averaged amplitude.
    """
    pr = PositionResult(position=0.0)
    pr.n_sides = len(sides)
    good = [s for s in sides if s.peak.accepted]
    pr.n_good_sides = len(good)
    pr.n_overridden = sum(1 for s in sides if s.peak.manual_override is not None)
    pr.n_suspect_sides = sum(1 for s in sides if s.peak.suspect)
    noise = [s.sigma for s in sides if np.isfinite(s.sigma)]
    pr.noise_floor = float(np.median(noise)) if noise else 0.0

    # Sawtooth amplitude: a per-position diagnostic of the laser sweep depth
    # reaching the detector, independent of the absorption fit.
    spans = [v for v in (ramp_span_volts(s) for s in sides) if np.isfinite(v)]
    if spans:
        pr.ramp_span_V = float(np.mean(spans))
        pr.ramp_span_err_V = float(np.std(spans, ddof=1)) if len(spans) > 1 else 0.0

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
                   progress=None, label: str = "") -> tuple[pd.DataFrame, list, dict]:
    """Stages 1-3: extract the absolute spatial absorption profile.

    Returns (per-position dataframe, PositionResults, manual overrides actually
    applied). The last one is empty unless `manual_overrides.json` in the output
    folder names sides present in this data.
    """
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

    # ── manual verdicts ──────────────────────────────────────────────────────
    # Applied to the final pass only, and never to the reference-linewidth pass
    # above: a side kept by hand must not redefine the matched-filter template.
    overrides = load_overrides(out) if config.use_manual_overrides else {}
    if overrides:
        _log(progress, f"Manual overrides: {len(overrides)} side verdict(s) set by hand "
                       f"in {out / 'manual_overrides.json'}")

    # ── combine sides, save plots & tables ───────────────────────────────────
    results: list[PositionResult] = []
    side_rows = []
    applied_overrides: dict[str, str] = {}
    for pos in sorted(traces):
        sides_here = final[pos]           # SideResults, in the order analyze_all built them
        # Re-pair each SideResult with its RampSide and a stable label. The order
        # here matches analyze_all exactly (same nested trace/side iteration).
        side_objs, stems = [], []
        for tr in traces[pos]:
            for side in tr.sides:
                side_objs.append(side)
                stems.append(f"{Path(tr.path).stem}:s{side.index}")

        for sr, stem in zip(sides_here, stems):
            verdict = overrides.get(stem)
            if verdict is None:
                continue
            apply_override(sr.peak, verdict)
            applied_overrides[stem] = verdict
            _log(progress, f"  override: {stem} forced {verdict.upper()} "
                           f"(auto verdict was {'accept' if sr.peak.auto_accepted else 'reject'})")

        pr = _combine_sides(sides_here, config)
        pr.position = pos
        pr.n_files = len(traces[pos])
        dcs = [s.dc_mean for s in side_objs if s.dc_mean is not None]
        pr.dc_mean = float(np.mean(dcs)) if dcs else None

        for side, sr, stem in zip(side_objs, sides_here, stems):
            pr.side_results.append((stem, sr))
            plotting.plot_side_fit(
                side, sr, str(dir_fits / side_image_name(stem)),
                title=f"{label} position {pos:g} — {stem}")
            side_rows.append({
                "position": pos, "side": stem, "image": side_image_name(stem),
                "amplitude": sr.peak.amplitude, "amplitude_err": sr.peak.amplitude_err,
                "center_ms": sr.peak.center * 1e3, "gamma_us": sr.peak.gamma * 1e6,
                "gamma_pinned_frac": sr.peak.gamma_pinned_frac,
                "snr": sr.peak.snr, "delta_aic": sr.peak.delta_aic,
                "accepted": sr.peak.accepted,
                "auto_accepted": sr.peak.auto_accepted,
                "suspect": sr.peak.suspect,
                "manual_override": sr.peak.manual_override or "",
                "reject_reasons": "; ".join(sr.peak.reject_reasons),
                "bg_fallback": sr.ramp.used_fallback,
                "ramp_span_V": ramp_span_volts(sr),
                "dc_mean_V": side.dc_mean,
            })
        results.append(pr)

    for pr in results:
        tag = "ok" if not pr.flagged else "FLAGGED (noise)"
        if pr.n_suspect_sides:
            tag += f", {pr.n_suspect_sides} suspect"
        if pr.n_overridden:
            tag += f", {pr.n_overridden} manual"
        _log(progress, f"  position {pr.position:g}: {pr.n_good_sides}/{pr.n_sides} sides good"
                       f"  -> A = {pr.amplitude:.4g} +- {pr.amplitude_err:.2g} %  [{tag}]")

    rows = [{
        "position": pr.position, "amplitude": pr.amplitude, "amplitude_err": pr.amplitude_err,
        "n_good_sides": pr.n_good_sides, "n_sides": pr.n_sides, "n_files": pr.n_files,
        "n_overridden": pr.n_overridden, "n_suspect_sides": pr.n_suspect_sides,
        "suspect": pr.n_suspect_sides > 0,
        "flagged": pr.flagged, "included": pr.include_in_profile,
        "noise_floor": pr.noise_floor, "dc_mean_V": pr.dc_mean,
        "ramp_span_V": pr.ramp_span_V, "ramp_span_err_V": pr.ramp_span_err_V,
    } for pr in results]
    df = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    df.to_csv(dir_prof / "absorption_profile.csv", index=False)
    pd.DataFrame(side_rows).to_csv(dir_prof / "side_details.csv", index=False)
    _log(progress, f"Wrote {dir_prof / 'absorption_profile.csv'} (+ side_details.csv)")

    plotting.plot_transmission(
        df, str(dir_prof / "transmission_vs_position.png"),
        title=f"{label} — CH2 DC transmission level vs position")
    plotting.plot_ramp_amplitude(
        df, config, str(dir_prof / "amplitude_vs_position.png"),
        title=f"{label} — CH3 sawtooth amplitude vs position")

    stale = sorted(set(overrides) - set(applied_overrides))
    if stale:
        _log(progress, f"  note: {len(stale)} override(s) match no side in this data "
                       f"(e.g. {stale[0]}) — left in place, not applied.")
    return df, results, applied_overrides


# ──────────────────────────────────────────────────────────────────────────────
def run_profile(df: pd.DataFrame, output_folder: str, config: AnalysisConfig,
                progress=None, label: str = "", n_overrides: int = 0) -> dict[str, ProfileFit]:
    """Fit the configured models to the spatial profile; save plots.

    `n_overrides` only annotates the figures — a profile built with hand-set side
    verdicts says so on its face, so a plot can never be mistaken for a purely
    automatic result.
    """
    dir_prof = Path(output_folder) / "03_profile"
    dir_prof.mkdir(parents=True, exist_ok=True)

    x = df["position"].to_numpy(dtype=float)
    y = df["amplitude"].to_numpy(dtype=float)
    e = df["amplitude_err"].to_numpy(dtype=float)
    included = df["included"].to_numpy(dtype=bool)
    accepted = ~df["flagged"].to_numpy(dtype=bool)
    suspect = (df["suspect"].to_numpy(dtype=bool) if "suspect" in df.columns
               else np.zeros(len(x), dtype=bool))

    fits: dict[str, ProfileFit] = {}
    for model in config.profile_models:
        pf = fit_profile(x, y, e, model, config, included=included,
                         accepted=accepted, suspect=suspect)
        fits[model] = pf
        plotting.plot_profile(pf, config, str(dir_prof / f"profile_{model}.png"),
                              dataset_label=f"{label} — {model} profile fit",
                              n_overrides=n_overrides)
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
    df, _, overrides = run_extraction(data_folder, str(out), config, progress, label)

    _log(progress, "=== Stage 4: spatial profile fits ===")
    fits = run_profile(df, str(out), config, progress, label, n_overrides=len(overrides))

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
                              dataset_label=f"{label} | {model}",
                              n_overrides=len(overrides))

    results = {"label": label, "data_folder": str(data_folder),
               "absorption_mode": "absolute",
               # Recorded so a set of numbers can never be read as purely
               # automatic when a side verdict was in fact set by hand.
               "manual_overrides": overrides,
               "zone1": {}, "flux": {}}
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
