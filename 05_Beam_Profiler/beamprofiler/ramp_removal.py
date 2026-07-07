"""
Automatic polynomial background (ramp) removal.

The photodetector signal on each ramp side is the laser-frequency sawtooth
(plus RC-filter phase-shift curvature when the signal-conditioning circuit is
in the chain) with a small absorption dip on top. The old workflow removed the
ramp by fitting a polynomial to hand-picked "wing" time windows; here the
absorption feature is located and masked automatically by iterative
sigma-clipping, so no per-setup time windows are needed:

    1. Fit a degree-d polynomial to all points.
    2. Mask points whose residual exceeds clip_sigma × (robust residual RMS).
       Contiguous masked runs are dilated slightly so the whole dip (not just
       its core) is excluded.
    3. Refit on unmasked points; repeat until the mask stabilises.

The returned noise level sigma (RMS of unmasked residuals) is used downstream
both to weight the Lorentzian fit and for the SNR-based noise rejection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RampFit:
    coeffs: np.ndarray        # polynomial coefficients (np.polyfit order)
    background: np.ndarray    # evaluated polynomial over the side
    subtracted: np.ndarray    # absorption signal = -(v - background); dips become peaks
    mask_feature: np.ndarray  # bool, True where points were excluded as "feature"
    sigma_noise: float        # RMS of background-region residuals


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Widen True runs of a boolean mask by n points on each side."""
    if n <= 0 or not mask.any():
        return mask
    idx = np.flatnonzero(mask)
    out = mask.copy()
    for i in idx:
        out[max(0, i - n): i + n + 1] = True
    return out


def subtract_ramp(
    t: np.ndarray,
    v: np.ndarray,
    degree: int = 2,
    clip_sigma: float = 2.5,
    clip_iterations: int = 6,
    known_feature_mask: np.ndarray | None = None,
) -> RampFit:
    """Fit and subtract the ramp background from one side.

    Parameters
    ----------
    t, v                : time [s] and signal [V] of one (trimmed) ramp side
    degree              : polynomial degree (1 = linear ramp, 2 = adds filter curvature)
    clip_sigma          : residual threshold for masking the absorption feature
    clip_iterations     : maximum sigma-clip iterations
    known_feature_mask  : optional pre-mask (True = exclude) — used on weak-signal
                          positions where the peak location is already known from
                          the reference fit, guaranteeing the dip never leaks into
                          the background fit even when it is below the noise.
    """
    # Work in centred/scaled time for a well-conditioned polynomial fit.
    t0 = 0.5 * (t[0] + t[-1])
    ts = (t - t0) / max(t[-1] - t[0], 1e-12)

    mask = np.zeros(len(t), dtype=bool)
    if known_feature_mask is not None:
        mask |= known_feature_mask

    coeffs = np.polyfit(ts, v, deg=degree)
    dilate_n = max(2, len(t) // 100)

    for _ in range(clip_iterations):
        good = ~mask
        # Guard: never fit on fewer than 25% of the points.
        if good.sum() < max(degree + 2, len(t) // 4):
            break
        coeffs = np.polyfit(ts[good], v[good], deg=degree)
        resid = v - np.polyval(coeffs, ts)
        # Robust noise estimate from the residuals of current background points
        # (MAD is insensitive to the dip itself).
        sigma = 1.4826 * np.median(np.abs(resid[good] - np.median(resid[good])))
        sigma = max(sigma, 1e-15)
        new_mask = np.abs(resid) > clip_sigma * sigma
        # The absorption dip is negative-going in V; clip both signs anyway to
        # also reject spikes/glitches.
        new_mask = _dilate(new_mask, dilate_n)
        if known_feature_mask is not None:
            new_mask |= known_feature_mask
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    background = np.polyval(coeffs, ts)
    resid = v - background
    good = ~mask
    sigma_noise = float(np.std(resid[good], ddof=degree + 1)) if good.sum() > degree + 2 else float("nan")

    # Sign convention: absorption reduces V, so flip to make the feature a
    # positive peak (matches the old analysis and all downstream fitting).
    subtracted = -resid

    return RampFit(
        coeffs=coeffs,
        background=background,
        subtracted=subtracted,
        mask_feature=mask,
        sigma_noise=sigma_noise,
    )
