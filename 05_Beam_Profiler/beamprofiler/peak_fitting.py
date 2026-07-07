"""
Lorentzian absorption-peak fitting with statistical noise rejection.

The probe crosses the atomic beam perpendicularly, so Doppler broadening is
strongly suppressed and the absorption feature is Lorentzian (thesis Sec. 4.4).
Each ramp side yields one background-subtracted spectrum, fit with

    L(t) = A / (1 + ((t - t0) / gamma)^2)  [+ offset]

where t is scan time (proportional to laser detuning for a linear ramp),
A is the peak relative absorption, and FWHM = 2*gamma.

Two problems with blind curve_fit motivated this module (thesis Sec. 5.3):

1. Initial conditions had to be hand-tuned on one strong peak. Here we do the
   same thing automatically: the pipeline first fits the *reference* (highest
   SNR) position unconstrained, then all other positions inherit the reference
   linewidth and peak position as initial guesses AND bounds. Physically the
   linewidth and line centre do not change with probe height, so weak-signal
   fits are strongly constrained.

2. At the beam edges the fitter locked onto noise, inflating the profile tails.
   Every fit is therefore validated before being believed:
     * SNR test        — fitted amplitude must exceed snr_min × background noise
     * ΔAIC test       — the peak model must beat a background-only (constant)
                         model by at least delta_aic_min in AIC
     * bound-pinning   — fits stuck at a parameter bound are suspect
   Sides that fail are marked rejected; positions with fewer than
   `min_good_sides` accepted sides are rejected as a whole and reported as
   amplitude 0 with a noise-derived upper bound (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .config import AnalysisConfig


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
def lorentzian(t, A, t0, gamma):
    return A / (1.0 + ((t - t0) / gamma) ** 2)


def lorentzian_offset(t, A, t0, gamma, offset):
    return lorentzian(t, A, t0, gamma) + offset


@dataclass
class PeakFit:
    """Result of fitting one ramp side."""
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
    accepted: bool = False
    reject_reasons: list = field(default_factory=list)
    popt: np.ndarray | None = None

    def curve(self, t: np.ndarray) -> np.ndarray:
        if self.popt is None:
            return np.zeros_like(t)
        f = lorentzian_offset if len(self.popt) == 4 else lorentzian
        return f(t, *self.popt)


@dataclass
class ReferencePeak:
    """Linewidth/centre template extracted from the strongest position."""
    gamma: float                  # median HWHM across the reference sides [s]
    # Peak centre per side, expressed as a FRACTION of the side duration
    # (centres in absolute time differ side to side; the fractional position
    # within a sweep is reproducible).
    center_frac: dict             # side index -> fraction in [0, 1]
    duration: float               # typical side duration [s]


# ──────────────────────────────────────────────────────────────────────────────
# Initial guesses
# ──────────────────────────────────────────────────────────────────────────────
def _smooth(y: np.ndarray, win: int) -> np.ndarray:
    win = max(3, win | 1)
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="same")


def initial_guess(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Estimate (A, t0, gamma) from the smoothed data.

    The smoothing window is ~2% of the side so genuine (broader) peaks survive
    while single-sample spikes wash out.
    """
    ys = _smooth(y, len(y) // 50)
    i_max = int(np.argmax(ys))
    A0 = float(ys[i_max] - np.median(ys))
    t0 = float(t[i_max])

    # Width estimate: walk out from the maximum to the half-max crossings.
    half = np.median(ys) + 0.5 * A0
    i_lo = i_max
    while i_lo > 0 and ys[i_lo] > half:
        i_lo -= 1
    i_hi = i_max
    while i_hi < len(ys) - 1 and ys[i_hi] > half:
        i_hi += 1
    fwhm = max(float(t[i_hi] - t[i_lo]), 4.0 * float(np.median(np.diff(t))))
    return max(A0, 1e-12), t0, fwhm / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Core fit
# ──────────────────────────────────────────────────────────────────────────────
def fit_side(
    t: np.ndarray,
    y: np.ndarray,
    sigma_noise: float,
    config: AnalysisConfig,
    reference: ReferencePeak | None = None,
    side_index: int | None = None,
) -> PeakFit:
    """Fit one background-subtracted ramp side with a Lorentzian.

    With `reference` given, the linewidth is bounded to
    gamma_ref × config.gamma_rel_bounds and the centre to a window of
    ±center_tolerance_frac × side-duration around the reference centre.
    """
    result = PeakFit()
    duration = float(t[-1] - t[0])
    if duration <= 0 or len(t) < 20 or not np.isfinite(sigma_noise) or sigma_noise <= 0:
        result.reject_reasons.append("bad input data")
        return result

    A0, t0_guess, gamma0 = initial_guess(t, y)

    # ── bounds ────────────────────────────────────────────────────────────────
    if reference is not None:
        g_lo = reference.gamma * config.gamma_rel_bounds[0]
        g_hi = reference.gamma * config.gamma_rel_bounds[1]
        gamma0 = float(np.clip(reference.gamma, g_lo, g_hi))
        if side_index in reference.center_frac:
            c_frac = reference.center_frac[side_index]
        else:  # unseen side: use mean of known fractions
            c_frac = float(np.mean(list(reference.center_frac.values())))
        c_ref = t[0] + c_frac * duration
        tol = config.center_tolerance_frac * duration
        c_lo, c_hi = c_ref - tol, c_ref + tol
        t0_guess = float(np.clip(t0_guess, c_lo, c_hi))
    else:
        # Unconstrained (reference) fit: keep physically sane limits — the peak
        # must sit inside the sweep and be narrower than half the sweep.
        g_lo, g_hi = 2.0 * float(np.median(np.diff(t))), duration / 2.0
        c_lo, c_hi = float(t[0]), float(t[-1])
        gamma0 = float(np.clip(gamma0, g_lo, g_hi))

    if config.fit_peak_offset:
        model = lorentzian_offset
        p0 = [A0, t0_guess, gamma0, float(np.median(y))]
        lb = [0.0, c_lo, g_lo, -np.inf]
        ub = [np.inf, c_hi, g_hi, np.inf]
    else:
        model = lorentzian
        p0 = [A0, t0_guess, gamma0]
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
        float(perr[0]), float(perr[1]), float(perr[2]),
    )
    result.offset = float(popt[3]) if config.fit_peak_offset else 0.0

    # ── goodness of fit & statistical validation ─────────────────────────────
    resid = y - model(t, *popt)
    dof = max(len(y) - len(popt), 1)
    chi2_peak = float(np.sum((resid / sigma_noise) ** 2))
    result.chi2_reduced = chi2_peak / dof

    # Null model: background only (constant at the mean).
    resid_null = y - np.mean(y)
    chi2_null = float(np.sum((resid_null / sigma_noise) ** 2))

    # AIC = chi² + 2k for Gaussian errors with known sigma.
    aic_peak = chi2_peak + 2 * len(popt)
    aic_null = chi2_null + 2 * 1
    result.delta_aic = aic_null - aic_peak       # >0 means the peak model is better
    result.snr = result.amplitude / sigma_noise

    reasons = []
    if result.snr < config.snr_min:
        reasons.append(f"SNR {result.snr:.1f} < {config.snr_min}")
    if result.delta_aic < config.delta_aic_min:
        reasons.append(f"dAIC {result.delta_aic:.1f} < {config.delta_aic_min}")
    # Pinned at a linewidth bound -> fit ran away toward noise or a spike.
    if reference is not None:
        if result.gamma <= g_lo * 1.02 or result.gamma >= g_hi * 0.98:
            reasons.append("linewidth pinned at bound")

    result.reject_reasons = reasons
    result.accepted = not reasons
    return result


def make_reference(side_fits: dict, sides: list) -> ReferencePeak | None:
    """Build the ReferencePeak template from accepted fits of the reference position.

    Parameters
    ----------
    side_fits : dict side_index -> PeakFit (from the unconstrained reference fit)
    sides     : list[RampSide] of the reference trace (for durations/fractions)
    """
    accepted = {i: f for i, f in side_fits.items() if f.accepted}
    if not accepted:
        return None
    gammas = [f.gamma for f in accepted.values()]
    durations, fracs = [], {}
    by_index = {s.index: s for s in sides}
    for i, f in accepted.items():
        s = by_index[i]
        d = float(s.t[-1] - s.t[0])
        durations.append(d)
        fracs[i] = float((f.center - s.t[0]) / d)
    return ReferencePeak(
        gamma=float(np.median(gammas)),
        center_frac=fracs,
        duration=float(np.median(durations)),
    )
