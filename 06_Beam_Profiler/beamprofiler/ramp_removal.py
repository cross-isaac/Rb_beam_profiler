"""
Background (ramp) removal — peak-aware corners.

Within one ramp side the CH3 signal is the linear laser-power sweep (a straight
side of the triangle wave) with a small absorption dip on top. The background is
that straight line, measured directly from the flat baseline at the ends of the
side ("corners"): average a small cluster of points at each end and draw the
line through the two cluster means.

Because the resonance in this apparatus is crossed near the ramp apex, the dip
often sits close to ONE end of the side — right where a corner cluster would be.
So the clusters are made PEAK-AWARE: once the peak has been located, any cluster
points inside the peak window are dropped before the line is fit, and if one end
is entirely covered by the peak the line falls back to every baseline point
outside the peak. A dip sitting on a ramp apex therefore cannot tilt the
background line.

The workflow is two-pass per side (see peak_fitting / pipeline):
    1. corners with no peak window  -> subtract -> locate the peak,
    2. corners excluding that peak  -> subtract -> fit the Lorentzian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RampFit:
    coeffs: np.ndarray        # line coefficients in centred/scaled time (np.polyfit order)
    background: np.ndarray    # evaluated line over the side
    subtracted: np.ndarray    # absorption signal = -(v - background); the dip becomes a peak
    anchor: np.ndarray        # bool, True where a point was used to pin the background line
    sigma_noise: float        # RMS of the anchor-region residuals
    used_fallback: bool = False  # True if an end was fully covered by the peak (line fit to all baseline)


def subtract_ramp_corners(
    t: np.ndarray,
    v: np.ndarray,
    cluster_fraction: float = 0.06,
    exclude: tuple[float, float] | None = None,
) -> RampFit:
    """Fit and subtract the linear background of one ramp side.

    Parameters
    ----------
    t, v             : time [s] and CH3 signal [V] of one (trimmed) side
    cluster_fraction : width of each end anchor cluster, as a fraction of the
                       side — the "how much of the edge is used" control
    exclude          : optional (t_lo, t_hi) peak window to keep out of the
                       anchor. When given, any corner-cluster point inside it is
                       dropped; if that empties an end, the line is fit to every
                       baseline point outside the window instead.
    """
    n = len(t)
    k = int(np.clip(round(cluster_fraction * n), 3, n // 3))

    # Anchor clusters = the k baseline points nearest each END that lie OUTSIDE
    # the peak window. For a central peak these are just the two ends (classic
    # corners); for a drifted, off-centre peak the cluster on the peak side
    # slides inward to the nearest clean baseline, so a near-apex dip never sits
    # under an anchor and the line is never pinned to the curved ramp end.
    if exclude is not None:
        keep = np.flatnonzero(~((t >= exclude[0]) & (t <= exclude[1])))
    else:
        keep = np.arange(n)
    if len(keep) < 6:                          # peak covers almost everything
        keep = np.arange(n)

    left = keep[:k]
    right = keep[-k:]
    anchor = np.zeros(n, dtype=bool)
    anchor[left] = True
    anchor[right] = True
    # Fallback flag: the two clusters overlap / touch, i.e. the peak left too
    # little clean baseline to bracket it (extreme near-apex case).
    used_fallback = exclude is not None and left[-1] >= right[0] - 1

    if anchor.sum() < 3:                       # degenerate: use everything
        anchor = np.ones(n, dtype=bool)

    # Centred/scaled time for a well-conditioned line fit.
    t0 = 0.5 * (t[0] + t[-1])
    span = max(t[-1] - t[0], 1e-12)
    ts = (t - t0) / span
    coeffs = np.polyfit(ts[anchor], v[anchor], deg=1)
    background = np.polyval(coeffs, ts)
    resid = v - background

    sigma_noise = (float(np.std(resid[anchor], ddof=2))
                   if anchor.sum() > 4 else float("nan"))

    return RampFit(
        coeffs=coeffs,
        background=background,
        subtracted=-resid,           # dip (negative in V) -> positive peak
        anchor=anchor,
        sigma_noise=sigma_noise,
        used_fallback=used_fallback,
    )
