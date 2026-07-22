"""
Spatial-profile fitting, divergence (Method A) and capture efficiency.

The per-position peak amplitudes A(Δy) form the transverse absorption profile of
the atomic beam. Following the thesis (Sec. 4.5) it is fit with both a Gaussian
and a Lorentzian (their relative quality is physically interesting — capillary-
array beams have non-Gaussian shoulders near the nozzle). A pseudo-Voigt model
is available as a tiebreaker: its mixing parameter eta interpolates Gaussian
(eta=0) -> Lorentzian (eta=1).

From each profile fit:
  * FWHM -> Method A divergence: theta_1/2 = atan[ (FWHM - D_nozzle)/2 / z ]
  * capture efficiency: propagate the profile to the growth substrate and
    integrate the normalized shape over the aperture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

from .config import AnalysisConfig

IN_TO_CM = 2.54
THOU_PER_IN = 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# Profile models — parameterized directly by FWHM.
# ──────────────────────────────────────────────────────────────────────────────
def gaussian_profile(x, A, x0, fwhm):
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return A * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))


def lorentzian_profile(x, A, x0, fwhm):
    gamma = fwhm / 2.0
    return A / (1.0 + ((x - x0) / gamma) ** 2)


def pseudo_voigt_profile(x, A, x0, fwhm, eta):
    return A * (eta * lorentzian_profile(x, 1.0, x0, fwhm)
                + (1.0 - eta) * gaussian_profile(x, 1.0, x0, fwhm))


_MODELS = {
    "gaussian": (gaussian_profile, ["A", "x0", "FWHM"]),
    "lorentzian": (lorentzian_profile, ["A", "x0", "FWHM"]),
    "pseudo_voigt": (pseudo_voigt_profile, ["A", "x0", "FWHM", "eta"]),
}


@dataclass
class ProfileFit:
    model: str
    popt: np.ndarray
    perr: np.ndarray
    param_names: list
    chi2_reduced: float
    r_squared: float
    fwhm_mm: float
    fwhm_mm_err: float
    theta_half_deg: float          # Method A half-angle divergence
    theta_half_deg_err: float
    capture_eff: float
    capture_eff_err: float
    eta: float | None = None
    x: np.ndarray = field(default=None)
    y: np.ndarray = field(default=None)
    y_err: np.ndarray = field(default=None)
    included: np.ndarray = field(default=None)   # bool mask of points used in the fit
    accepted: np.ndarray = field(default=None)   # bool mask of noise-validated points (styling)
    suspect: np.ndarray = field(default=None)    # accepted-but-flagged points, flag mode (styling)

    @property
    def center(self) -> float:
        return float(self.popt[1])

    def curve(self, x: np.ndarray) -> np.ndarray:
        f, _ = _MODELS[self.model]
        return f(x, *self.popt)


# ──────────────────────────────────────────────────────────────────────────────
def _initial_profile_guess(x, y):
    A0 = float(np.max(y))
    x0 = float(x[np.argmax(y)])
    above = x[y > 0.5 * A0]
    fwhm0 = float(above.max() - above.min()) if len(above) > 1 else float(np.ptp(x) / 4.0)
    return A0, x0, max(fwhm0, float(np.min(np.diff(np.sort(x)))))


def fit_profile(x, y, y_err, model, config, included=None, accepted=None,
                suspect=None) -> ProfileFit:
    """Fit one model to the spatial profile A(Δy).

    `suspect` only styles the plot (amber "review me" points); suspect points are
    still accepted, so they are part of the fit like any other accepted point.
    """
    if included is None:
        included = np.ones(len(x), dtype=bool)
    if accepted is None:
        accepted = included.copy()
    if suspect is None:
        suspect = np.zeros(len(x), dtype=bool)
    xf, yf, ef = x[included], y[included], y_err[included]
    ef = np.where(ef > 0, ef, np.max(ef) if np.max(ef) > 0 else 1.0)

    func, names = _MODELS[model]
    A0, x00, fwhm0 = _initial_profile_guess(xf, yf)

    xs = np.sort(np.unique(xf))
    fwhm_min = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0

    p0 = [A0, x00, max(fwhm0, fwhm_min)]
    lb = [0.0, xf.min() - np.ptp(xf), fwhm_min]
    ub = [np.inf, xf.max() + np.ptp(xf), 10.0 * np.ptp(xf)]
    if model == "pseudo_voigt":
        p0 += [0.5]; lb += [0.0]; ub += [1.0]
    if config.profile_offset:
        func_fit = lambda xx, *p: func(xx, *p[:-1]) + p[-1]  # noqa: E731
        names = names + ["offset"]
        p0 += [0.0]; lb += [-np.inf]; ub += [np.inf]
    else:
        func_fit = func

    popt, pcov = curve_fit(func_fit, xf, yf, p0=p0, sigma=ef, absolute_sigma=True,
                           bounds=(lb, ub), method="trf", maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    fit_y = func_fit(xf, *popt)
    resid = yf - fit_y
    dof = max(len(yf) - len(popt), 1)
    chi2_red = float(np.sum((resid / ef) ** 2) / dof)
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan

    upm = config.position_units_per_mm
    fwhm_mm = float(popt[2]) / upm
    fwhm_mm_err = float(perr[2]) / upm
    theta, theta_err = divergence_method_a(float(popt[2]), float(perr[2]), config)

    eta = float(popt[3]) if model == "pseudo_voigt" else None
    P, P_err = capture_efficiency(model, float(popt[2]), float(perr[2]),
                                  theta, theta_err, config, eta=eta)

    return ProfileFit(
        model=model, popt=popt, perr=perr, param_names=names,
        chi2_reduced=chi2_red, r_squared=r2,
        fwhm_mm=fwhm_mm, fwhm_mm_err=fwhm_mm_err,
        theta_half_deg=theta, theta_half_deg_err=theta_err,
        capture_eff=P, capture_eff_err=P_err, eta=eta,
        x=x, y=y, y_err=y_err, included=included, accepted=accepted, suspect=suspect,
    )


# ──────────────────────────────────────────────────────────────────────────────
def divergence_method_a(fwhm_thou, fwhm_thou_err, config) -> tuple[float, float]:
    """Single-zone geometric half-angle (thesis Eq. 4.11-4.12), in degrees."""
    z = config.z_probe_in
    fwhm_in = fwhm_thou / THOU_PER_IN
    u = ((fwhm_in - config.nozzle_diameter_in) / 2.0) / z
    theta = math.atan(u)
    dtheta = (1.0 / (1.0 + u ** 2)) / (2.0 * z) * (fwhm_thou_err / THOU_PER_IN)
    return math.degrees(theta), math.degrees(abs(dtheta))


def _normalized_profile_integral(model, fwhm_cm, a_cm, eta) -> float:
    func = _MODELS[model][0]
    if model == "pseudo_voigt":
        shape = lambda yy: func(yy, 1.0, 0.0, fwhm_cm, eta)   # noqa: E731
    else:
        shape = lambda yy: func(yy, 1.0, 0.0, fwhm_cm)        # noqa: E731
    norm, _ = quad(shape, -np.inf, np.inf)
    inside, _ = quad(shape, -a_cm, a_cm)
    return inside / norm if norm > 0 else float("nan")


def capture_efficiency(model, fwhm_thou, fwhm_thou_err, theta_deg, theta_deg_err,
                       config, eta=None) -> tuple[float, float]:
    """Fraction of the beam passing the substrate aperture (thesis Eq. 4.14-4.15)."""
    a = config.aperture_radius_cm
    D_cm = (config.z_substrate_in - config.z_probe_in) * IN_TO_CM

    def P_of(fwhm_t, th_deg):
        fwhm_cm = fwhm_t / THOU_PER_IN * IN_TO_CM
        fwhm_sub = fwhm_cm + 2.0 * D_cm * math.tan(math.radians(th_deg))
        return _normalized_profile_integral(model, max(fwhm_sub, 1e-9), a, eta)

    P = P_of(fwhm_thou, theta_deg)
    dP_dw = (P_of(fwhm_thou + fwhm_thou_err, theta_deg) - P) if fwhm_thou_err > 0 else 0.0
    dP_dt = (P_of(fwhm_thou, theta_deg + theta_deg_err) - P) if theta_deg_err > 0 else 0.0
    return P, math.sqrt(dP_dw ** 2 + dP_dt ** 2)
