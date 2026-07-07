"""
All figure generation for the pipeline.

Every stage of the analysis writes a PNG so the whole chain can be audited
after the fact (and browsed in the GUI):

  01_raw_traces/    full scope trace with the detected ramp-side boundaries
  02_side_fits/     per side: background fit (top) and Lorentzian fit (bottom)
  03_profile/       spatial profile A(Δy) with each model fit + parameter table
  04_flux/          absolute-absorption fits of the unattenuated trace
  summary_*.png     final column-density profile + full results table
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: the GUI displays the saved PNGs instead
import matplotlib.pyplot as plt
import numpy as np

from .config import AnalysisConfig

# Consistent styling across all figures.
DATA_COLOR = "#e08214"     # orange — raw points
FIT_COLOR = "#2c7fb8"      # blue  — fits
MASK_COLOR = "#66c2a5"     # green — masked/background regions
BAD_COLOR = "#d7301f"      # red   — rejected


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
def plot_trace_segmentation(trace, path: str, title: str = "") -> str:
    """Full raw trace with detected side boundaries (stage 1)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(trace.t * 1e3, trace.v, lw=0.3, color=DATA_COLOR, label="signal")
    if trace.sync is not None:
        ax2 = ax.twinx()
        ax2.plot(trace.t * 1e3, trace.sync, lw=0.4, color="gray", alpha=0.5, label="sync")
        ax2.set_ylabel("Sync [V]", color="gray", fontsize=9)
        ax2.tick_params(labelsize=8, colors="gray")
    for s in trace.sides:
        ax.axvline(s.t_start * 1e3, color=FIT_COLOR, lw=0.8, ls="--", alpha=0.8)
        ax.text(0.5 * (s.t_start + s.t_end) * 1e3, ax.get_ylim()[1],
                f"side {s.index}", ha="center", va="bottom", fontsize=8, color=FIT_COLOR)
    if trace.sides:
        ax.axvline(trace.sides[-1].t_end * 1e3, color=FIT_COLOR, lw=0.8, ls="--", alpha=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Signal [V]")
    ax.set_title(title or os.path.basename(trace.path), fontsize=9, color="gray")
    return _save(fig, path)


def plot_side_fit(side, ramp_fit, peak_fit, path: str, title: str = "",
                  y_label: str = "Relative absorption [arb. u.]",
                  y_fitted=None) -> str:
    """Two-panel stage plot for one ramp side (stages 2+3).

    Top:    raw side with the fitted polynomial background; the automatically
            masked feature region is shaded.
    Bottom: the signal that was actually fit (pass `y_fitted` when it differs
            from the plain subtracted voltage, e.g. percent absorption for
            flux traces) with the Lorentzian fit and the accept/reject verdict.
    """
    t_ms = side.t * 1e3
    y_bottom = ramp_fit.subtracted if y_fitted is None else y_fitted
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"hspace": 0.08})

    ax1.plot(t_ms, side.v, ".", ms=1.5, color=DATA_COLOR, label="raw data")
    ax1.plot(t_ms, ramp_fit.background, color=MASK_COLOR, lw=1.5,
             label=f"background (deg {len(ramp_fit.coeffs) - 1})")
    if ramp_fit.mask_feature.any():
        m = ramp_fit.mask_feature
        ax1.fill_between(t_ms, *ax1.get_ylim(), where=m, color=MASK_COLOR,
                         alpha=0.15, label="masked (feature)")
    ax1.set_ylabel("Signal [V]", fontsize=10)
    ax1.legend(fontsize=8, loc="best")
    ax1.set_title(title, fontsize=9, color="gray")

    ax2.plot(t_ms, y_bottom, ".", ms=1.5, color=DATA_COLOR, label="data − background")
    if peak_fit.popt is not None:
        color = FIT_COLOR if peak_fit.accepted else BAD_COLOR
        ax2.plot(t_ms, peak_fit.curve(side.t), color=color, lw=1.8, label="Lorentzian fit")
        verdict = "ACCEPTED" if peak_fit.accepted else "REJECTED: " + "; ".join(peak_fit.reject_reasons)
        info = (f"A = {peak_fit.amplitude:.3g} ± {peak_fit.amplitude_err:.2g}   "
                f"SNR = {peak_fit.snr:.1f}   ΔAIC = {peak_fit.delta_aic:.0f}\n{verdict}")
    else:
        color = BAD_COLOR
        info = "FIT FAILED: " + "; ".join(peak_fit.reject_reasons)
    ax2.text(0.02, 0.96, info, transform=ax2.transAxes, fontsize=8,
             va="top", color=color,
             bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.85))
    ax2.set_xlabel("Laser scan time [ms]", fontsize=10)
    ax2.set_ylabel(y_label, fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    return _save(fig, path)


# ──────────────────────────────────────────────────────────────────────────────
def plot_profile(pf, config: AnalysisConfig, path: str, dataset_label: str = "",
                 rescale: float = 1.0, y_label: str = "Relative absorption [arb. u.]") -> str:
    """Spatial profile A(Δy) with one model fit and its parameter table."""
    upm = config.position_units_per_mm
    x0 = pf.center

    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={"width_ratios": [2, 1]})
    if dataset_label:
        fig.suptitle(dataset_label, fontsize=9, color="gray")

    good = pf.accepted if pf.accepted is not None else pf.included
    bad = ~good
    ax.errorbar((pf.x[good] - x0) / upm, rescale * pf.y[good], yerr=rescale * pf.y_err[good],
                fmt="x", ms=5, capsize=2, ls="None", color=FIT_COLOR, label="data")
    if bad.any():
        ax.errorbar((pf.x[bad] - x0) / upm, rescale * pf.y[bad], yerr=rescale * pf.y_err[bad],
                    fmt="o", ms=4, capsize=2, ls="None", color=BAD_COLOR,
                    alpha=0.6, label="rejected (noise)")
    x_fit = np.linspace(pf.x.min() - 0.05 * np.ptp(pf.x), pf.x.max() + 0.05 * np.ptp(pf.x), 1500)
    ax.plot((x_fit - x0) / upm, rescale * pf.curve(x_fit), color=FIT_COLOR, lw=1.8,
            alpha=0.9, label=f"{pf.model.replace('_', '-')} fit")
    ax.set_xlabel(r"$\Delta y$ [mm]", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(which="major", ls="--", alpha=0.4)
    ax.legend(fontsize=10)

    rows = [
        ["Fit model", pf.model.replace("_", "-")],
        [r"Reduced $\chi^2$", f"{pf.chi2_reduced:.3f}"],
        [r"$R^2$", f"{pf.r_squared:.4f}"],
        ["FWHM [mm]", f"{pf.fwhm_mm:.2f} ± {pf.fwhm_mm_err:.2f}"],
        [r"$\theta_{1/2}$ [deg]", f"{pf.theta_half_deg:.3f} ± {pf.theta_half_deg_err:.3f}"],
        ["Capture eff. [%]", f"{pf.capture_eff * 100:.1f} ± {pf.capture_eff_err * 100:.1f}"],
        ["Probe dist. [in]", f"{config.z_probe_in:.2f}"],
    ]
    if pf.eta is not None:
        rows.insert(1, [r"$\eta$ (0=Gauss, 1=Lorentz)", f"{pf.eta:.2f}"])

    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=["Parameter", "Value"],
                     loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)
    for j in range(2):
        tbl[0, j].set_facecolor("#41b6c4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    return _save(fig, path)


def plot_summary(pf, flux_result, config: AnalysisConfig, path: str,
                 dataset_label: str = "") -> str:
    """Final figure: column-density profile + complete results table."""
    upm = config.position_units_per_mm
    x0 = pf.center
    rescale = flux_result.rescale_factor if flux_result else 1.0
    y_label = (r"Rb column density [cm$^{-2}$]" if flux_result
               else "Relative absorption [arb. u.]")

    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11.5, 5.5),
                                   gridspec_kw={"width_ratios": [2, 1]})
    if dataset_label:
        fig.suptitle(dataset_label, fontsize=9, color="gray")

    inc = pf.included
    ax.errorbar((pf.x[inc] - x0) / upm, rescale * pf.y[inc], yerr=rescale * pf.y_err[inc],
                fmt="x", ms=5, capsize=2, ls="None", color=FIT_COLOR, label="data")
    x_fit = np.linspace(pf.x.min() - 0.05 * np.ptp(pf.x), pf.x.max() + 0.05 * np.ptp(pf.x), 1500)
    ax.plot((x_fit - x0) / upm, rescale * pf.curve(x_fit), color=FIT_COLOR, lw=1.8,
            label=f"{pf.model.replace('_', '-')} fit")
    ax.set_xlabel(r"$\Delta y$ [mm]", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(which="major", ls="--", alpha=0.4)
    ax.legend(fontsize=10)

    rows = [
        ["Fit model", pf.model.replace("_", "-")],
        [r"Reduced $\chi^2$", f"{pf.chi2_reduced:.3f}"],
        [r"$R^2$", f"{pf.r_squared:.4f}"],
        ["FWHM [mm]", f"{pf.fwhm_mm:.2f} ± {pf.fwhm_mm_err:.2f}"],
        [r"$\theta_{1/2}$ [deg]", f"{pf.theta_half_deg:.3f} ± {pf.theta_half_deg_err:.3f}"],
        ["Capture eff. [%]", f"{pf.capture_eff * 100:.1f} ± {pf.capture_eff_err * 100:.1f}"],
        ["Probe dist. [cm]", f"{config.z_probe_in * 2.54:.2f}"],
    ]
    if flux_result:
        rows += [
            ["Rb flux [atoms/s]", f"{flux_result.rb_flux_atoms_s:.2e}"],
            [r"$n_\mathrm{Rb}$ (beam) [cm$^{-3}$]", f"{flux_result.number_density_cm3:.1e}"],
        ]
        if flux_result.dopant_density_cm3 is not None:
            rows.append([r"$\rho_\mathrm{Rb}$ (crystal) [cm$^{-3}$]",
                         f"{flux_result.dopant_density_cm3:.2e}"])

    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=["Parameter", "Value"],
                     loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)
    for j in range(2):
        tbl[0, j].set_facecolor("#41b6c4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    return _save(fig, path)
