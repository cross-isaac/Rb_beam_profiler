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

`find_linear_core` handles a second, systematic artifact seen in the raw
(op-amp-bypassed) data: optics imperfections bend the first/last ~10-20% of
each nominally linear ramp side by several times the noise. Those curved ends
cannot be represented by the background polynomial, so the largest central
window in which the background is genuinely polynomial is detected and both
the background and peak fits are restricted to it.
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


def subtract_ramp_corners(
    t: np.ndarray,
    v: np.ndarray,
    cluster_fraction: float = 0.05,
) -> RampFit:
    """Background from the triangle-wave amplitude itself ("corners" method).

    A small cluster of points at each end of the side — i.e. near the two ramp
    turn points — is averaged, and the background is simply the straight line
    through the two cluster means. Nothing is fit iteratively and nothing is
    masked, so there are no per-side decisions to jitter between sweeps: the
    line is pinned the same way on every side. (The side has already had the
    fixed segmentation edge-trim applied, so the clusters sit just off the
    rounded apex itself.)

    The noise estimate comes from the scatter of the cluster points about
    their own means — by construction free of the absorption dip.
    """
    n = len(t)
    k = int(np.clip(round(cluster_fraction * n), 3, n // 3))
    t1, v1 = float(np.mean(t[:k])), float(np.mean(v[:k]))
    t2, v2 = float(np.mean(t[-k:])), float(np.mean(v[-k:]))
    slope = (v2 - v1) / max(t2 - t1, 1e-12)
    background = v1 + slope * (t - t1)
    resid = v - background

    scatter = np.concatenate([v[:k] - np.mean(v[:k]), v[-k:] - np.mean(v[-k:])])
    sigma_noise = float(np.std(scatter, ddof=2)) if len(scatter) > 4 else float("nan")

    anchors = np.zeros(n, dtype=bool)
    anchors[:k] = True
    anchors[-k:] = True

    # Express the line in the same scaled-time convention as np.polyfit-based
    # methods so downstream consumers (plot labels, residual evaluation) agree.
    t0 = 0.5 * (t[0] + t[-1])
    span = max(t[-1] - t[0], 1e-12)
    coeffs = np.array([slope * span, v1 + slope * (t0 - t1)])

    return RampFit(
        coeffs=coeffs,
        background=background,
        subtracted=-resid,
        mask_feature=~anchors,
        sigma_noise=sigma_noise,
    )


def subtract_ramp_wings(
    t: np.ndarray,
    v: np.ndarray,
    center: float,
    inner_halfwidth: float,
    degree: int = 1,
) -> RampFit:
    """Background from two point clusters flanking the peak ("wings" method).

    Instead of sigma-clip masking, the polynomial is fit ONLY to the points
    more than `inner_halfwidth` away from the (known or bootstrapped) peak
    centre — i.e. the clusters near the start and end of the sweep window.
    This is geometric and cannot over-mask: the anchor regions are always the
    same, and everything between them is interpolated under the peak.

    If either wing would end up with too few points (peak near a window edge),
    the inner half-width is shrunk until both wings hold at least ~8% of the
    side, so the interpolation always has support on both sides.
    """
    n = len(t)
    min_pts = max(degree + 2, n // 12)
    inner = float(inner_halfwidth)
    wings = (t < center - inner) | (t > center + inner)
    for _ in range(8):
        left = int(np.count_nonzero(t < center - inner))
        right = int(np.count_nonzero(t > center + inner))
        if left >= min_pts and right >= min_pts:
            break
        inner *= 0.75
        wings = (t < center - inner) | (t > center + inner)
    else:
        # Degenerate geometry — fall back to the sigma-clip method.
        return subtract_ramp(t, v, degree=degree)

    t0 = 0.5 * (t[0] + t[-1])
    ts = (t - t0) / max(t[-1] - t[0], 1e-12)
    coeffs = np.polyfit(ts[wings], v[wings], deg=degree)
    background = np.polyval(coeffs, ts)
    resid = v - background
    sigma_noise = float(np.std(resid[wings], ddof=degree + 1)) \
        if wings.sum() > degree + 2 else float("nan")

    return RampFit(
        coeffs=coeffs,
        background=background,
        subtracted=-resid,
        mask_feature=~wings,
        sigma_noise=sigma_noise,
    )


def find_linear_core(
    t: np.ndarray,
    v: np.ndarray,
    degree: int = 1,
    clip_sigma: float = 2.5,
    clip_iterations: int = 6,
    known_feature_mask: np.ndarray | None = None,
    max_trim_fraction: float = 0.30,
    resid_sigma: float = 2.0,
) -> np.ndarray:
    """Boolean mask of the largest central window with a truly polynomial background.

    The background polynomial is anchored on the CENTRAL half of the side (the
    part verified linear; a whole-side fit would tilt to accommodate the curved
    ends and hide them). Residuals against that anchored background are then
    smoothed over `win` points and walked inward from each end: points are
    trimmed until `win` CONSECUTIVE points sit within `resid_sigma` × the noise
    of a win-point mean — i.e. until the deviation is provably noise, not a
    systematic bow or etalon wiggle. Each end trims at most `max_trim_fraction`
    of the side, so a peak sitting mid-sweep can never be eaten; the known
    absorption window (when given) is exempt from the walk entirely. A second
    pass refits the background on the surviving core for refinement.

    Returns a bool array (True = keep) over the full side.
    """
    n = len(t)
    full = np.ones(n, dtype=bool)
    if n < 60:
        return full
    max_trim_n = int(max_trim_fraction * n)
    win = max(5, n // 30)             # smoothing window for the residual walk
    kernel = np.ones(win)
    # Edge-normalized moving average (plain 'same' convolution zero-pads and
    # would suppress exactly the end regions being tested).
    norm = np.convolve(np.ones(n), kernel, mode="same")

    anchor = np.zeros(n, dtype=bool)
    anchor[int(0.25 * n): int(0.75 * n)] = True
    core = full.copy()

    for _ in range(2):
        fit_on = anchor & core
        rf = subtract_ramp(t[fit_on], v[fit_on], degree=degree, clip_sigma=clip_sigma,
                           clip_iterations=clip_iterations,
                           known_feature_mask=(known_feature_mask[fit_on]
                                               if known_feature_mask is not None else None))
        if not np.isfinite(rf.sigma_noise) or rf.sigma_noise <= 0:
            return core
        # Residuals over the FULL side relative to the anchored background
        # (same time scaling subtract_ramp used internally).
        t0 = 0.5 * (t[fit_on][0] + t[fit_on][-1])
        span = max(t[fit_on][-1] - t[fit_on][0], 1e-12)
        resid = v - np.polyval(rf.coeffs, (t - t0) / span)
        sm = np.convolve(resid, kernel, mode="same") / norm
        walk = np.abs(sm)
        # Feature region (known + sigma-clip detected): real signal, not
        # background curvature — exempt from the walk and the noise estimate.
        feature = np.zeros(n, dtype=bool)
        feature[fit_on] = rf.mask_feature
        if known_feature_mask is not None:
            feature |= known_feature_mask
        walk[feature] = 0.0
        # Noise scale of the SMOOTHED residual, measured empirically on the
        # quiet part of the anchor region (the CH3 traces carry a correlated
        # etalon ripple, so a win-point mean does NOT shrink like sqrt(win)
        # white noise — the scale must come from the data).
        quiet = fit_on & ~feature
        if quiet.sum() < 10:
            return core
        sm_q = sm[quiet]
        sigma_sm = 1.4826 * float(np.median(np.abs(sm_q - np.median(sm_q))))
        sigma_sm = max(sigma_sm, 0.25 * rf.sigma_noise / np.sqrt(win))
        thresh = resid_sigma * sigma_sm

        def first_clean(indices):
            """First kept index scanning inward: needs `win` consecutive quiet points."""
            run, limit = 0, max_trim_n + win
            for k, i in enumerate(indices):
                if k >= limit:
                    break
                if walk[i] <= thresh:
                    run += 1
                    if run == win:
                        return k - win + 1
                else:
                    run = 0
            return max_trim_n

        lo = first_clean(range(n))
        hi = n - 1 - first_clean(range(n - 1, -1, -1))
        if hi - lo < max(2 * win, n // 4):     # pathological: keep everything
            return core
        new_core = np.zeros(n, dtype=bool)
        new_core[lo:hi + 1] = True
        # Refinement pass anchors on the detected core; keep the wider anchor
        # if the first pass trimmed the anchor region itself away.
        if np.array_equal(new_core, core):
            break
        core = new_core
    return core


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
