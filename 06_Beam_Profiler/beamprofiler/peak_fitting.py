"""
Per-side absorption-peak analysis: peak-aware background, matched-filter
localization, Lorentzian fit, and noise rejection.

The probe crosses the atomic beam perpendicularly, so Doppler broadening is
suppressed and the absorption feature is Lorentzian (thesis Sec. 4.4):

    L(t) = A / (1 + ((t - t0) / gamma)^2)  [+ offset]

t is scan time (proportional to detuning for a linear ramp), A is the peak
absorption, and FWHM = 2*gamma.

Why this differs from a blind curve_fit — and from the v1 pipeline:

* The laser drifts across a run and is nudged back by hand mid-scan, so the peak
  wanders over almost the whole sweep (measured: fractional position 0.23-0.87)
  with no smooth trend to exploit. Every side is therefore localized
  INDEPENDENTLY, over its whole length, by a matched filter: a Lorentzian
  template (at the reference linewidth, plus a small width bank) is slid across
  the side and the best-scoring position wins. This uses the known lineshape for
  noise-robustness while searching everywhere for drift-robustness.

* The linewidth genuinely broadens toward the beam edges (more divergent /
  thermal atoms), so the reference width is used only to LOCATE the peak; the
  refined fit lets the width float over a wide range and never rejects a fit for
  being broad.

Validation (per side): the fit is accepted only if
    SNR  = A / background-noise RMS         >= snr_min,
    dAIC = AIC(background-only) - AIC(peak) >= delta_aic_min, and
    gamma / upper linewidth bound           <  linewidth_runaway_fraction
(the last one catches a Lorentzian that ran away and ate the ramp curvature
instead of fitting a line; set the fraction to 0 to switch that test off).

A verdict can also be set by hand from the GUI's image browser — see
side_overrides.py. Manual verdicts are recorded separately (`manual_override`,
`auto_accepted`) so a figure or CSV always shows what the automatic tests said.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .config import AnalysisConfig
from .ramp_removal import RampFit, subtract_ramp_corners


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
def lorentzian(t, A, t0, gamma):
    return A / (1.0 + ((t - t0) / gamma) ** 2)


def lorentzian_offset(t, A, t0, gamma, offset):
    return lorentzian(t, A, t0, gamma) + offset


@dataclass
class PeakFit:
    amplitude: float = 0.0
    amplitude_err: float = 0.0
    center: float = 0.0            # peak position in scan time [s]
    center_err: float = 0.0
    gamma: float = 0.0             # HWHM in scan time [s]
    gamma_err: float = 0.0
    offset: float = 0.0
    chi2_reduced: float = np.nan
    snr: float = 0.0
    delta_aic: float = 0.0
    gamma_pinned_frac: float = 0.0  # fitted HWHM / upper linewidth bound (1.0 = pinned)
    accepted: bool = False          # EFFECTIVE verdict (manual override wins)
    auto_accepted: bool = False     # what the automatic tests alone concluded
    manual_override: str | None = None   # "accept" / "reject" / None, set by side_overrides
    reject_reasons: list = field(default_factory=list)
    popt: np.ndarray | None = None

    @property
    def verdict_label(self) -> str:
        """Human-readable verdict, naming a manual override when one is in force."""
        if self.manual_override is None:
            return "ACCEPTED" if self.accepted else "REJECTED"
        was = "would auto-accept" if self.auto_accepted else "would auto-reject"
        return ("ACCEPTED" if self.accepted else "REJECTED") + f" — MANUAL OVERRIDE ({was})"

    def curve(self, t: np.ndarray) -> np.ndarray:
        if self.popt is None:
            return np.zeros_like(t)
        f = lorentzian_offset if len(self.popt) == 4 else lorentzian
        return f(t, *self.popt)


@dataclass
class SideResult:
    """Everything produced for one ramp side."""
    ramp: RampFit          # final peak-aware background (in volts, over the full side)
    peak: PeakFit
    y: np.ndarray          # final background-subtracted signal in fit units (% absorption)
    sigma: float           # noise RMS in the same units as y
    center_t: float        # matched-filter peak location [s]
    exclude: tuple | None  # peak window kept out of the background anchor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _smooth(y: np.ndarray, win: int) -> np.ndarray:
    win = max(3, win | 1)
    return np.convolve(y, np.ones(win) / win, mode="same")


def _estimate_width(t: np.ndarray, y: np.ndarray) -> float:
    """Rough HWHM [s] of the dominant positive feature (bootstrap, no reference)."""
    ys = _smooth(y, max(3, len(y) // 50))
    i = int(np.argmax(ys))
    A = float(ys[i] - np.median(ys))
    half = np.median(ys) + 0.5 * A
    lo, hi = i, i
    while lo > 0 and ys[lo] > half:
        lo -= 1
    while hi < len(ys) - 1 and ys[hi] > half:
        hi += 1
    fwhm = max(float(t[hi] - t[lo]), 4.0 * float(np.median(np.diff(t))))
    return fwhm / 2.0


def matched_filter_localize(
    t: np.ndarray, y: np.ndarray, gamma_templates, margin_frac: float,
) -> tuple[float, float, float]:
    """Slide Lorentzian templates over the side; return (t0, gamma, amplitude).

    At each candidate centre and template width the best (amplitude, offset) is
    solved in closed form and scored by the matched-filter detection statistic
    (amplitude x template norm). Only positive amplitudes (real dips) count.
    """
    n = len(t)
    duration = float(t[-1] - t[0])
    margin = margin_frac * duration
    grid = np.linspace(t[0] + margin, t[-1] - margin, min(160, n))

    best = (float(t[np.argmax(_smooth(y, max(3, n // 50)))]),
            float(gamma_templates[0]), 0.0)
    best_score = -np.inf
    y0 = y - np.mean(y)
    for g in gamma_templates:
        g = float(max(g, 2.0 * np.median(np.diff(t))))
        for t0 in grid:
            s = 1.0 / (1.0 + ((t - t0) / g) ** 2)
            s0 = s - np.mean(s)
            ss = float(np.dot(s0, s0))
            if ss <= 0:
                continue
            A = float(np.dot(y0, s0) / ss)
            if A <= 0:
                continue
            score = A * np.sqrt(ss)          # matched-filter detection statistic
            if score > best_score:
                best_score = score
                best = (float(t0), g, A)
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Refine + validate
# ──────────────────────────────────────────────────────────────────────────────
def _refine_and_validate(
    t: np.ndarray, y: np.ndarray, sigma_noise: float, config: AnalysisConfig,
    t0_seed: float, gamma_scale: float, amp_seed: float,
) -> PeakFit:
    result = PeakFit()
    duration = float(t[-1] - t[0])
    if duration <= 0 or len(t) < 20 or not np.isfinite(sigma_noise) or sigma_noise <= 0:
        result.reject_reasons.append("bad input data")
        return result

    dt = float(np.median(np.diff(t)))
    # ── linewidth bounds around a ROBUST scale (reference HWHM, or the half-max
    # estimate in the bootstrap pass) — NOT the matched-filter template width,
    # which can be pulled wide by curved ramp ends. gamma_rel_bounds is generous
    # enough to allow real edge broadening but tight enough that a broad
    # Lorentzian cannot run away and swallow the background curvature.
    gamma_scale = max(gamma_scale, 2.0 * dt)
    g_lo = max(gamma_scale * config.gamma_rel_bounds[0], 2.0 * dt)
    g_hi = min(gamma_scale * config.gamma_rel_bounds[1], 0.5 * duration)
    gamma0 = float(np.clip(gamma_scale, g_lo, g_hi))

    # ── centre bounds: a generous window around the matched-filter location ──────
    tol = max(config.center_refine_gammas * gamma0, 5.0 * dt)
    c_lo = max(t0_seed - tol, float(t[0]))
    c_hi = min(t0_seed + tol, float(t[-1]))
    t0_guess = float(np.clip(t0_seed, c_lo, c_hi))

    if config.fit_peak_offset:
        model = lorentzian_offset
        p0 = [max(amp_seed, 1e-9), t0_guess, gamma0, float(np.median(y))]
        lb = [0.0, c_lo, g_lo, -np.inf]
        ub = [np.inf, c_hi, g_hi, np.inf]
    else:
        model = lorentzian
        p0 = [max(amp_seed, 1e-9), t0_guess, gamma0]
        lb = [0.0, c_lo, g_lo]
        ub = [np.inf, c_hi, g_hi]

    try:
        popt, pcov = curve_fit(
            model, t, y, p0=p0,
            sigma=np.full_like(y, sigma_noise), absolute_sigma=True,
            bounds=(lb, ub), method="trf", maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:  # noqa: BLE001 — any optimizer failure = rejection
        result.reject_reasons.append(f"fit failed: {exc}")
        return result

    result.popt = popt
    result.amplitude, result.center, result.gamma = float(popt[0]), float(popt[1]), float(popt[2])
    result.amplitude_err, result.center_err, result.gamma_err = (
        float(perr[0]), float(perr[1]), float(perr[2]))
    result.offset = float(popt[3]) if config.fit_peak_offset else 0.0

    resid = y - model(t, *popt)
    dof = max(len(y) - len(popt), 1)
    chi2_peak = float(np.sum((resid / sigma_noise) ** 2))
    result.chi2_reduced = chi2_peak / dof
    chi2_null = float(np.sum(((y - np.mean(y)) / sigma_noise) ** 2))
    result.delta_aic = (chi2_null + 2 * 1) - (chi2_peak + 2 * len(popt))
    result.snr = result.amplitude / sigma_noise

    reasons = []
    if result.snr < config.snr_min:
        reasons.append(f"SNR {result.snr:.1f} < {config.snr_min}")
    if result.delta_aic < config.delta_aic_min:
        reasons.append(f"dAIC {result.delta_aic:.1f} < {config.delta_aic_min}")
    # A fit pinned at the UPPER linewidth bound is a runaway: a broad Lorentzian
    # absorbing ramp-end curvature (inflating A and SNR), not a real line. This
    # is distinct from the modest broadening allowed within the bounds, which is
    # legitimate. (Pinning at the lower bound is fine — narrow is physical.)
    # How close to the bound counts as "pinned" is the sensitivity knob
    # `linewidth_runaway_fraction`; 0 disables the test entirely.
    result.gamma_pinned_frac = result.gamma / g_hi if g_hi > 0 else 0.0
    frac = config.linewidth_runaway_fraction
    if frac > 0 and result.gamma_pinned_frac >= frac:
        reasons.append(f"linewidth ran away ({result.gamma_pinned_frac:.2f} of the "
                       f"{g_hi * 1e6:.0f} us bound, limit {frac:.2f})")
    result.reject_reasons = reasons
    result.accepted = not reasons
    result.auto_accepted = result.accepted
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Full per-side analysis (two-pass peak-aware corners + fit)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_side(
    t: np.ndarray, v: np.ndarray,
    dc_mean: float | None, dc_err: float | None,
    config: AnalysisConfig, ref_gamma: float | None,
) -> SideResult:
    """Background-subtract, locate, and fit the absorption peak on one side.

    Pass 1 removes a plain-corners background and locates the peak by matched
    filter; pass 2 removes a corners background that excludes that peak window
    (so a near-apex dip cannot tilt the line) and the Lorentzian is fitted to
    the result, normalized to percent absorption by the CH2 DC level.
    """
    denom = None
    if dc_mean is not None:
        denom = dc_mean - config.dark_offset_V
    if denom is None or denom <= 0:
        denom = 1.0                                   # fall back to raw volts

    # ── pass 1: plain corners -> locate ──────────────────────────────────────
    ramp0 = subtract_ramp_corners(t, v, config.corner_cluster_fraction)
    y0 = 100.0 * ramp0.subtracted / denom

    # The width SCALE for the fit bounds is the reference HWHM when known, else
    # a self-contained half-max estimate. The matched-filter template width is
    # used only to detect/localize the peak, never to set the fit's linewidth.
    if ref_gamma is not None and ref_gamma > 0:
        scale = ref_gamma
        widths = [w * ref_gamma for w in config.match_widths_rel]
    else:
        scale = _estimate_width(t, y0)
        widths = [0.5 * scale, scale, 2.0 * scale]
    t0_mf, g_mf, A_mf = matched_filter_localize(
        t, y0, widths, config.search_margin_fraction)

    # ── pass 2: peak-aware corners -> fit ────────────────────────────────────
    win = (t0_mf - config.peak_exclude_gammas * scale,
           t0_mf + config.peak_exclude_gammas * scale)
    ramp = subtract_ramp_corners(t, v, config.corner_cluster_fraction, exclude=win)
    y = 100.0 * ramp.subtracted / denom
    sigma = 100.0 * ramp.sigma_noise / denom

    peak = _refine_and_validate(
        t, y, sigma, config,
        t0_seed=t0_mf, gamma_scale=scale, amp_seed=100.0 * A_mf / denom)

    # Fold the DC-normalization uncertainty into the amplitude error.
    if peak.popt is not None and dc_mean is not None and dc_err is not None:
        rel_dc = dc_err / denom
        peak.amplitude_err = float(np.hypot(peak.amplitude_err, peak.amplitude * rel_dc))

    return SideResult(ramp=ramp, peak=peak, y=y, sigma=sigma,
                      center_t=t0_mf, exclude=win)


def reference_gamma_from(results: dict[float, list[SideResult]]) -> tuple[float | None, float | None]:
    """Reference linewidth for the matched-filter template scale.

    The physical HWHM is the same at every position (set by the transition, not
    the probe height), so it is best measured on the CLEANEST fits: the reference
    is the median HWHM of the highest-SNR accepted sides run-wide. Using only the
    strong, well-resolved sides keeps the template narrow and physical — a
    reference inflated by weak, broad edge fits widens the peak-exclude window and
    the linewidth bounds enough to let broad Lorentzians run away elsewhere.
    Also returns the strongest position (highest median accepted-side SNR).
    """
    accepted = [s for sides in results.values() for s in sides if s.peak.accepted]
    if not accepted:
        return None, None
    accepted.sort(key=lambda s: s.peak.snr, reverse=True)
    n_top = max(6, len(accepted) // 4)
    ref_gamma = float(np.median([s.peak.gamma for s in accepted[:n_top]]))

    best_pos, best_snr = None, -np.inf
    for pos, sides in results.items():
        acc = [s for s in sides if s.peak.accepted]
        if acc:
            med_snr = float(np.median([s.peak.snr for s in acc]))
            if med_snr > best_snr:
                best_snr, best_pos = med_snr, pos
    return best_pos, ref_gamma
