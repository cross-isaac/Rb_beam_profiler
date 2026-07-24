"""
Absolute flux and dopant-density calculations (thesis Sec. 4.6).

Run-format data measures a true fractional absorption at every position
(CH3 dip / CH2 DC level), so the fitted profile A(y) [%] IS the absorption map —
no calibration trace is needed. Point-wise:

    fractional absorption A  ->  OD = -ln(1 - A)             (Beer-Lambert)
    column density N_col = OD / (sigma_eff * f_F3 * P_85)    [cm^-2, all Rb]
    integrate N_col across the beam -> N_slice               [cm^-1]
    flux  = N_slice * v_mean,  v_mean = sqrt(8 kB T / pi m)  [atoms/s]

For dual-species operation the Rb:Ar flux ratio maps onto the dopant density in
the grown crystal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .config import AnalysisConfig
from .profile_analysis import ProfileFit, _MODELS

KB = 1.380649e-23                   # J/K
M_RB = 85.4678 * 1.66053906660e-27  # kg, natural-abundance Rb atomic mass
NA = 6.02214076e23                  # 1/mol
IN_TO_CM = 2.54
THOU_PER_IN = 1000.0


@dataclass
class FluxResult:
    absorption_frac: float        # peak fractional absorption (profile centre)
    optical_depth: float
    n_col_cm2: float              # total-Rb column density at the profile centre
    n_slice_cm1: float            # transverse integral of the column density
    v_mean_cm_s: float
    rb_flux_atoms_s: float
    number_density_cm3: float     # rough volume density (N_col / beam width)
    rescale_factor: float         # multiply the % profile by this -> column density
    reference_position: float     # stage units (profile centre)
    ar_flux_atoms_s: float | None = None
    dopant_density_cm3: float | None = None


def flux_from_absolute_profile(profile: ProfileFit, config: AnalysisConfig) -> FluxResult:
    """Absolute Rb flux straight from a profile fitted in percent absorption."""
    func = _MODELS[profile.model][0]
    # SHAPE PARAMETERS ONLY. If the profile was fitted with a constant pedestal
    # (config.profile_offset), it is deliberately excluded here: a pedestal is a
    # baseline that does not fall off in the wings, so integrating it over the
    # transverse coordinate diverges, and it is not beam. Flux and capture
    # efficiency therefore both describe the peak ABOVE the pedestal, while the
    # plotted curve includes it.
    args = tuple(profile.popt[:4]) if profile.model == "pseudo_voigt" else tuple(profile.popt[:3])
    thou_to_cm = IN_TO_CM / THOU_PER_IN
    scale = 1.0 / (config.sigma_eff_cm2 * config.f_F3 * config.p_Rb85)

    A_peak = float(func(profile.center, *args)) / 100.0
    od_peak = -math.log(max(1.0 - A_peak, 1e-12))
    n_col_peak = od_peak * scale

    def n_col_cm(y_cm):
        A = func(y_cm / thou_to_cm, *args) / 100.0
        A = np.clip(A, 0.0, 1.0 - 1e-12)
        return -np.log1p(-A) * scale

    n_slice, _ = quad(n_col_cm, -np.inf, np.inf)

    T_K = 273.15 + config.oven_temp_C
    v_mean = math.sqrt(8.0 * KB * T_K / (math.pi * M_RB)) * 100.0   # cm/s
    rb_flux = n_slice * v_mean

    width_cm = profile.fwhm_mm / 10.0
    n_vol = n_col_peak / width_cm if width_cm > 0 else float("nan")

    result = FluxResult(
        absorption_frac=A_peak, optical_depth=od_peak, n_col_cm2=n_col_peak,
        n_slice_cm1=n_slice, v_mean_cm_s=v_mean, rb_flux_atoms_s=rb_flux,
        number_density_cm3=n_vol,
        rescale_factor=n_col_peak / max(A_peak * 100.0, 1e-12),
        reference_position=profile.center,
    )
    if config.ar_flow_sccm is not None:
        ar_flux, dopant = dopant_density(config.ar_flow_sccm, rb_flux,
                                         config.ar_density_crystal_kg_m3)
        result.ar_flux_atoms_s = ar_flux
        result.dopant_density_cm3 = dopant
    return result


def dopant_density(ar_flow_sccm, rb_flux, ar_density_crystal=1768.0) -> tuple[float, float]:
    """Expected Rb dopant density in the Ar crystal from the flux ratio.

    Returns (Ar flux [atoms/s], Rb dopant density [atoms/cm^3]).
    """
    R = 8.3145            # J/(mol K)
    Z = 0.99906           # Ar compressibility at STP
    P = 101325.0          # Pa
    M_AR = 39.948e-3      # kg/mol
    T0 = 273.15           # K

    flow_m3_s = ar_flow_sccm * 1e-6 / 60.0
    mass_flow = flow_m3_s * (P * M_AR) / (Z * R * T0)      # kg/s
    ar_flux = mass_flow / M_AR * NA                        # atoms/s
    ar_number_per_cm3 = (ar_density_crystal / M_AR) * NA * 1e-6
    return ar_flux, (rb_flux / ar_flux) * ar_number_per_cm3
