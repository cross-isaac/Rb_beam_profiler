"""
All figure generation for the pipeline. Every stage writes a PNG so the whole
chain can be audited (and browsed in the GUI):

  01_raw_traces/   full scope trace with the detected ramp-side boundaries
  02_side_fits/    per side: background fit (top) and Lorentzian fit (bottom)
  03_profile/      spatial profile A(Δy) with each model fit + parameter table,
                   plus two per-position diagnostics: the CH2 DC transmission
                   level and the CH3 sawtooth amplitude
  summary_*.png    final column-density profile + full results table

Figures built with hand-set side verdicts are stamped as such (see
side_overrides.py) — purple box on the side image, footnote on the profile and
summary figures.
"""

from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")  # headless: the GUI displays the saved PNGs
import matplotlib.pyplot as plt
import numpy as np

from .config import AnalysisConfig

DATA_COLOR = "#e08214"     # orange — raw points
FIT_COLOR = "#2c7fb8"      # blue  — fits
MASK_COLOR = "#66c2a5"     # green — background anchors
BAD_COLOR = "#d7301f"      # red   — rejected
OVERRIDE_COLOR = "#762a83" # purple — verdict set by hand, not by the tests


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _override_footnote(fig, n_overrides: int) -> None:
    """Stamp a figure that was built with hand-set side verdicts."""
    if not n_overrides:
        return
    fig.text(0.01, 0.005,
             f"{n_overrides} side verdict(s) set by hand (manual_overrides.json) — "
             f"not a purely automatic result",
             fontsize=7.5, color=OVERRIDE_COLOR, ha="left", va="bottom")


# ──────────────────────────────────────────────────────────────────────────────
def plot_trace_segmentation(trace, path: str, title: str = "") -> str:
    """Full raw trace with detected side boundaries (stage 1)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(trace.t * 1e3, trace.v, lw=0.3, color=DATA_COLOR, label="signal (CH3)")
    if trace.sync is not None:
        ax2 = ax.twinx()
        ax2.plot(trace.t * 1e3, trace.sync, lw=0.4, color="gray", alpha=0.5)
        ax2.set_ylabel("Sync (CH4) [V]", color="gray", fontsize=9)
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


def plot_side_fit(side, side_result, path: str, title: str = "",
                  y_label: str = "Absorption [%]") -> str:
    """Two-panel stage plot for one ramp side.

    Top:    raw side (V) with the peak-aware corners background. The points used
            to anchor the line are green; the located peak window kept out of the
            anchor is shaded; the matched-filter peak location is a dashed line.
    Bottom: the background-subtracted signal (% absorption) with the Lorentzian
            fit and the accept/reject verdict.
    """
    ramp = side_result.ramp
    peak = side_result.peak
    t_ms = side.t * 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"hspace": 0.08})

    # ── top: background ──────────────────────────────────────────────────────
    ax1.plot(t_ms, side.v, ".", ms=1.5, color=DATA_COLOR, label="raw data (CH3)")
    ax1.plot(t_ms, ramp.background, color=MASK_COLOR, lw=1.5, label="background (line)")
    if ramp.anchor.any():
        ax1.plot(t_ms[ramp.anchor], side.v[ramp.anchor], ".", ms=2.5,
                 color=MASK_COLOR, label="anchor points" + (" (fallback)" if ramp.used_fallback else ""))
    if side_result.exclude is not None:
        lo, hi = side_result.exclude
        ax1.axvspan(lo * 1e3, hi * 1e3, color="gray", alpha=0.12, label="peak window (excluded)")
    ax1.axvline(side_result.center_t * 1e3, color=BAD_COLOR, lw=0.9, ls=":",
                alpha=0.7, label="located peak")
    ax1.set_ylabel("Signal [V]", fontsize=10)
    ax1.legend(fontsize=8, loc="best")
    ax1.set_title(title, fontsize=9, color="gray")

    # ── bottom: subtracted + fit ─────────────────────────────────────────────
    ax2.plot(t_ms, side_result.y, ".", ms=1.5, color=DATA_COLOR, label="data − background")
    if peak.popt is not None:
        color = FIT_COLOR if peak.accepted else BAD_COLOR
        ax2.plot(t_ms, peak.curve(side.t), color=color, lw=1.8, label="Lorentzian fit")
        verdict = peak.verdict_label
        if peak.reject_reasons:
            verdict += ": " + "; ".join(peak.reject_reasons)
        info = (f"A = {peak.amplitude:.3g} ± {peak.amplitude_err:.2g} %   "
                f"SNR = {peak.snr:.1f}   ΔAIC = {peak.delta_aic:.0f}   "
                f"HWHM = {peak.gamma * 1e6:.0f} µs "
                f"({peak.gamma_pinned_frac:.2f} of the runaway bound)\n"
                + textwrap.fill(verdict, width=95))
    else:
        color = BAD_COLOR
        info = textwrap.fill("FIT FAILED: " + "; ".join(peak.reject_reasons), width=95)
    # The verdict goes UNDER the axes, not inside them: reject reasons can run
    # long, and a box floating over the data hid whichever part of the trace
    # explains the verdict. A hand-set verdict gets a heavier, differently
    # coloured box so it is obvious at a glance which images are not automatic.
    edge = OVERRIDE_COLOR if peak.manual_override is not None else color
    lw = 2.2 if peak.manual_override is not None else 1.0
    ax2.set_xlabel("Laser scan time [ms]", fontsize=10)
    ax2.set_ylabel(y_label, fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.text(0.0, -0.20, info, transform=ax2.transAxes, fontsize=8, va="top", color=color,
             bbox=dict(boxstyle="round", fc="white", ec=edge, lw=lw, alpha=0.9))
    return _save(fig, path)


def plot_transmission(df, path: str, title: str = "") -> str:
    """CH2 DC transmission level vs stage position (window-transmission diagnostic)."""
    d = df.dropna(subset=["dc_mean_V"]).sort_values("position")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d["position"], d["dc_mean_V"], "o-", ms=5, color=FIT_COLOR)
    ax.set_xlabel("Stage position [thou]", fontsize=11)
    ax.set_ylabel("CH2 DC level [V]", fontsize=11)
    ax.set_title(title, fontsize=9, color="gray")
    ax.grid(ls="--", alpha=0.4)
    ax.minorticks_on()
    return _save(fig, path)


def plot_ramp_amplitude(df, config: AnalysisConfig, path: str, title: str = "") -> str:
    """Sawtooth (laser-power sweep) amplitude vs stage position.

    Companion diagnostic to transmission_vs_position.png. The background under
    each ramp side is the straight sweep, so its end-to-end span is the triangle
    wave's trough-to-crest amplitude on CH3; the plotted point is the mean over
    that position's sides, the error bar their scatter.

    Top    — span in volts, as measured on the fitted (edge-trimmed) side. It is
             not extrapolated to the full sweep: the trimmed-off part is the ramp
             apex, which is curved, so extrapolating a straight line through it
             would overstate the amplitude.
    Bottom — the same span divided by the CH2 DC level, i.e. the fractional depth
             of the sweep reaching the detector. If CH3 is simply the AC-coupled
             copy of CH2 (the assumption the absolute-absorption normalization
             already rests on), this ratio is a property of the laser sweep and
             should be FLAT across the scan. A slope or step here means something
             other than uniform attenuation changed mid-run — detector gain, a
             scope range change, or the sweep amplitude itself — and the absolute
             absorption scale at those positions deserves a second look.
    """
    d = df.sort_values("position")
    has_dc = "dc_mean_V" in d.columns and d["dc_mean_V"].notna().any()
    fig, axes = plt.subplots(2 if has_dc else 1, 1, figsize=(7, 6.4 if has_dc else 4),
                             sharex=True, squeeze=False,
                             gridspec_kw={"hspace": 0.12})
    ax = axes[0][0]

    err = d["ramp_span_err_V"] if "ramp_span_err_V" in d.columns else None
    ax.errorbar(d["position"], d["ramp_span_V"], yerr=err, fmt="o-", ms=5, capsize=3,
                color=FIT_COLOR)
    ax.set_ylabel("Sawtooth amplitude [V]\n(CH3, peak-to-peak)", fontsize=10)
    ax.set_title(title, fontsize=9, color="gray")
    ax.grid(ls="--", alpha=0.4)
    ax.minorticks_on()
    trim = 2.0 * config.edge_trim_fraction
    ax.text(0.99, 0.03, f"measured over the fitted side ({trim:.0%} trimmed at the apexes)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="gray")

    if has_dc:
        ax2 = axes[1][0]
        frac = 100.0 * d["ramp_span_V"] / d["dc_mean_V"]
        ax2.plot(d["position"], frac, "s-", ms=4.5, color=DATA_COLOR)
        ax2.axhline(float(np.median(frac)), color="gray", ls="--", lw=0.9,
                    label=f"median {float(np.median(frac)):.2f} %")
        ax2.set_ylabel("Sweep depth\n(CH3 span / CH2 DC) [%]", fontsize=10)
        ax2.grid(ls="--", alpha=0.4)
        ax2.minorticks_on()
        ax2.legend(fontsize=8)
        ax2.set_xlabel("Stage position [thou]", fontsize=11)
    else:
        ax.set_xlabel("Stage position [thou]", fontsize=11)
    return _save(fig, path)


# ──────────────────────────────────────────────────────────────────────────────
def plot_profile(pf, config: AnalysisConfig, path: str, dataset_label: str = "",
                 rescale: float = 1.0, y_label: str = "Absorption [%]",
                 n_overrides: int = 0) -> str:
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
                    fmt="o", ms=4, capsize=2, ls="None", color=BAD_COLOR, alpha=0.6,
                    label="noise-dominated")
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
    _param_table(ax_t, rows)
    fig.tight_layout()
    _override_footnote(fig, n_overrides)
    return _save(fig, path)


def plot_summary(pf, flux_result, config: AnalysisConfig, path: str,
                 dataset_label: str = "", n_overrides: int = 0) -> str:
    """Final figure: column-density profile + complete results table."""
    upm = config.position_units_per_mm
    x0 = pf.center
    rescale = flux_result.rescale_factor if flux_result else 1.0
    y_label = (r"Rb column density [cm$^{-2}$]" if flux_result else "Absorption [%]")

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
    _param_table(ax_t, rows)
    fig.tight_layout()
    _override_footnote(fig, n_overrides)
    return _save(fig, path)


def _param_table(ax_t, rows):
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=["Parameter", "Value"],
                     loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)
    for j in range(2):
        tbl[0, j].set_facecolor("#41b6c4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
