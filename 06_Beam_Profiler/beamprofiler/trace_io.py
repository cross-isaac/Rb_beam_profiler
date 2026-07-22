"""
Loading and segmenting raw oscilloscope traces (run format only).

Data layout — the acquisition runner writes one file per scan step:

    # timestamp: 2026-07-20 16:48:33.944
    # step: 4
    # target_in: 0.50000
    time_s,CH2_V,CH3_V,CH4_V
    -0.003281,1.04987,-0.00039192,2.20058
    ...

  * CH2 — raw DC photodetector level (no amplifier). Sets the absolute
          transmission scale per side, so window-transmission differences
          across the viewport are normalized out.
  * CH3 — AC-coupled copy of the same signal at fine vertical scale; this is
          what the background removal and peak fits run on.
  * CH4 — scan-controller sync square wave.
  * Probe position comes from `# target_in` (inches -> thou).

Segmentation: the sync square wave (CH4) gives the sweep PERIOD, but its edges
are phase-shifted from the real sawtooth turning points. So only the period is
taken from the sync; the turning-point phase is measured from the signal itself
by folding the trace at the period and fitting an ideal triangle wave with a
robust loss (a deep absorption dip acts as a localized outlier and cannot pull
the corners). Each resulting side is one up- or down-sweep through resonance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import AnalysisConfig

# Column roles in the run format.
TIME_COL = "time_s"
DC_COL = "CH2_V"       # raw DC photodetector level
SIGNAL_COL = "CH3_V"   # AC-coupled signal — this is what gets fitted
SYNC_COL = "CH4_V"
IN_TO_THOU = 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RampSide:
    """One half-sweep (up or down) of the laser frequency ramp."""
    index: int                # 1-based side number within the trace
    t: np.ndarray             # time [s], bin-averaged
    v: np.ndarray             # CH3 signal [V], bin-averaged
    t_start: float            # untrimmed boundaries in the original trace
    t_end: float
    dc_mean: float | None = None   # mean CH2 DC transmission level [V] over this side
    dc_err: float | None = None    # its uncertainty (scatter, floored at 1/2 ADC LSB)


@dataclass
class Trace:
    """A full scope acquisition for one probe position."""
    path: str
    t: np.ndarray             # full time vector [s]
    v: np.ndarray             # CH3 signal [V]
    sync: np.ndarray | None   # CH4 sync [V], or None if absent
    dc: np.ndarray | None = None                 # raw CH2 DC channel
    meta: dict = field(default_factory=dict)     # '# key: value' header metadata
    position_thou: float | None = None           # stage position from metadata [thou]
    sides: list = field(default_factory=list)    # list[RampSide], filled by segment()


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────
def read_metadata(path: str) -> dict:
    """Parse the leading '# key: value' comment lines of a run-format CSV."""
    meta: dict = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if ":" in line:
                k, v = line[1:].split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


def _bin_average(t: np.ndarray, v: np.ndarray, n_bins: int):
    """Average (t, v) into at most n_bins equal-time bins."""
    if len(t) <= n_bins:
        return t, v
    edges = np.linspace(t[0], t[-1], n_bins + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    t_out = np.bincount(idx, weights=t, minlength=n_bins)
    v_out = np.bincount(idx, weights=v, minlength=n_bins)
    good = counts > 0
    return t_out[good] / counts[good], v_out[good] / counts[good]


def load_trace(path: str, config: AnalysisConfig) -> Trace:
    """Load one run-format scope CSV."""
    meta = read_metadata(path)
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    missing = {TIME_COL, SIGNAL_COL} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: not a run-format file — missing column(s) {sorted(missing)}. "
            f"Found {list(df.columns)}. This tool reads only the acquisition-runner "
            f"format (time_s,CH2_V,CH3_V,CH4_V with a '# target_in' header)."
        )

    t = df[TIME_COL].to_numpy(dtype=float)
    v = df[SIGNAL_COL].to_numpy(dtype=float)
    dc = df[DC_COL].to_numpy(dtype=float) if DC_COL in df.columns else None
    sync = df[SYNC_COL].to_numpy(dtype=float) if SYNC_COL in df.columns else None

    order = np.argsort(t, kind="stable")   # guard against repeats/reversals
    t, v = t[order], v[order]
    if dc is not None:
        dc = dc[order]
    if sync is not None:
        sync = sync[order]

    position_thou = None
    if "target_in" in meta:
        try:
            position_thou = float(meta["target_in"]) * IN_TO_THOU
        except ValueError:
            pass

    return Trace(path=str(path), t=t, v=v, sync=sync, dc=dc,
                 meta=meta, position_thou=position_thou)


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────────────────────────────────────
def _half_period_from_sync(t: np.ndarray, sync: np.ndarray) -> float | None:
    """Half-period of the sweep from the sync square wave (median edge spacing)."""
    lo, hi = np.percentile(sync, [5, 95])
    if hi - lo < 0.1:                      # sync channel flat/absent
        return None
    state = sync > 0.5 * (lo + hi)
    edges = np.flatnonzero(np.diff(state.astype(int)) != 0)
    if len(edges) < 2:
        return None
    te = t[edges]
    # Debounce glitchy syncs: keep an edge only if well separated from the last.
    min_sep = 0.2 * float(np.max(np.diff(te))) if len(te) > 1 else 0.0
    kept = [float(te[0])]
    for x in te[1:]:
        if float(x) - kept[-1] >= min_sep:
            kept.append(float(x))
    te = np.asarray(kept)
    if len(te) < 2:
        return None
    if len(te) >= 3:
        # Lag-2 spacing = one full period, immune to a non-50% duty cycle.
        return float(np.median(te[2:] - te[:-2])) / 2.0
    return float(te[1] - te[0])


def _half_period_auto(t: np.ndarray, v: np.ndarray) -> float:
    """Half-period from the autocorrelation of the signal (sync-less fallback)."""
    n = min(len(v), 20000)
    idx = np.linspace(0, len(v) - 1, n).astype(int)
    vv = v[idx] - np.mean(v[idx])
    tt = t[idx]
    ac = np.correlate(vv, vv, mode="full")[n - 1:]
    ac /= ac[0]
    i = 1
    while i < n - 1 and ac[i] > 0:
        i += 1
    if i >= n - 2:
        return (t[-1] - t[0]) / 4.0
    j = i + int(np.argmax(ac[i:]))
    return float(tt[j] - tt[0]) / 2.0


def _triangle(t, t0, period, phi0, duty, amp, off):
    """Ideal (possibly asymmetric) triangle wave. phi0 in [0,1): phase of the
    minimum turning point; duty: rising fraction of the period."""
    x = np.mod((t - t0) / period - phi0, 1.0)
    up = x < duty
    y = np.empty_like(x)
    y[up] = -0.5 + x[up] / duty
    y[~up] = 0.5 - (x[~up] - duty) / (1.0 - duty)
    return amp * y + off


def _fit_triangle_full(t, v, period_seed):
    """Fit a triangle wave to the whole trace with the PERIOD free.

    Fitting period, phase, and duty jointly (rather than folding at a fixed,
    possibly-slightly-wrong sync period) keeps the turning points aligned across
    the whole record, so the ramp sides come out even. The tiny absorption dip
    is a localized outlier under the soft-L1 loss and cannot pull the corners.
    Returns (period, phi0, duty) with phi0 the phase of a minimum turn point.
    """
    from scipy.optimize import least_squares

    t0 = float(t[0])
    amp0 = float(np.ptp(v))
    off0 = float(np.mean(v))
    phi0_init = float(np.mod((np.argmin(v) / len(v)), 1.0))
    p0 = [period_seed, phi0_init, 0.5, amp0, off0]
    lb = [0.7 * period_seed, -1.0, 0.2, 0.3 * amp0, -np.inf]
    ub = [1.3 * period_seed, 2.0, 0.8, 3.0 * amp0, np.inf]
    res = least_squares(
        lambda p: _triangle(t, t0, *p) - v, p0, bounds=(lb, ub),
        loss="soft_l1", f_scale=max(amp0 * 0.05, 1e-9), max_nfev=4000,
    )
    period, phi0, duty = float(res.x[0]), float(res.x[1] % 1.0), float(res.x[2])
    return period, phi0, duty


def _period_seed(t, v, sync) -> float:
    t_half = _half_period_from_sync(t, sync) if sync is not None else None
    if t_half is None:
        t_half = _half_period_auto(t, v)
    return 2.0 * t_half


def _fit_triangle_phase(t, v, period, duty, seed_phi0):
    """Fit only the phase of a triangle wave with period & duty held fixed.

    Used once the global period/duty are known: for each trace only the trigger
    phase varies, so this one-parameter fit is very robust.
    """
    from scipy.optimize import least_squares

    t0 = float(t[0])
    amp0 = float(np.ptp(v))
    off0 = float(np.mean(v))

    def resid(p):
        phi0, amp, off = p
        return _triangle(t, t0, period, phi0, duty, amp, off) - v

    res = least_squares(resid, [seed_phi0, amp0, off0], loss="soft_l1",
                        f_scale=max(amp0 * 0.05, 1e-9), max_nfev=2000)
    return float(res.x[0] % 1.0)


def _bounds_from(t, period, phi0, duty) -> list[float]:
    """Turn-point boundaries of a triangle wave over the record."""
    span0, span1 = float(t[0]), float(t[-1])
    inner: list[float] = []
    for phase in (phi0, (phi0 + duty) % 1.0):
        b = span0 + phase * period
        while b <= span0:
            b += period
        while b < span1:
            inner.append(float(b))
            b += period
    inner.sort()
    return [span0] + inner + [span1]


def _build_sides(trace: Trace, bounds, period, duty, config) -> Trace:
    """Turn boundaries into RampSides (edge-trim, bin-average, drop partial ends)."""
    d_up, d_down = duty * period, (1.0 - duty) * period
    dc_lsb = 0.0
    if trace.dc is not None:
        levels = np.unique(trace.dc)
        if len(levels) > 1:
            dc_lsb = float(np.min(np.diff(levels)))

    sides: list[RampSide] = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        dur = t1 - t0
        sel_dir = (trace.t >= t0) & (trace.t <= t1)
        if sel_dir.sum() >= 2:
            slope = np.polyfit(trace.t[sel_dir], trace.v[sel_dir], 1)[0]
            expected = d_up if slope > 0 else d_down
        else:
            expected = min(d_up, d_down)
        if dur < config.min_side_fraction * expected:      # truncated end ramp
            continue
        trim = config.edge_trim_fraction * dur
        sel = (trace.t >= t0 + trim) & (trace.t <= t1 - trim)
        if sel.sum() < 50:
            continue
        ts, vs = _bin_average(trace.t[sel], trace.v[sel], config.downsample_points_per_side)
        dc_mean = dc_err = None
        if trace.dc is not None:
            dseg = trace.dc[sel]
            dc_mean = float(np.mean(dseg))
            dc_err = max(float(np.std(dseg) / np.sqrt(len(dseg))), 0.5 * dc_lsb)
        sides.append(RampSide(index=len(sides) + 1, t=ts, v=vs, t_start=t0, t_end=t1,
                              dc_mean=dc_mean, dc_err=dc_err))
    trace.sides = sides
    return trace


def segment(trace: Trace, config: AnalysisConfig,
            period: float | None = None, duty: float | None = None) -> Trace:
    """Split one trace into RampSides.

    With `period`/`duty` given (the shared sweep geometry of the run), only the
    trigger phase is fitted per trace — far more robust than a free per-trace
    fit. Otherwise the period, phase and duty are all fitted from this trace.
    """
    seed = _period_seed(trace.t, trace.v, trace.sync)
    if period is None or duty is None:
        period, phi0, duty = _fit_triangle_full(trace.t, trace.v, seed)
    else:
        _, phi0_seed, _ = _fit_triangle_full(trace.t, trace.v, seed)
        phi0 = _fit_triangle_phase(trace.t, trace.v, period, duty, phi0_seed)
    return _build_sides(trace, _bounds_from(trace.t, period, phi0, duty), period, duty, config)


def segment_all(traces: list[Trace], config: AnalysisConfig) -> tuple[float, float]:
    """Segment a whole run consistently.

    The sweep period and duty cycle are fixed instrument settings for the run,
    so they are fitted per trace, reduced to a robust median, and then applied to
    every trace (phase-only per trace). This prevents the occasional bad
    per-trace period fit from mis-segmenting a single file. Returns the global
    (period, duty).
    """
    periods, duties = [], []
    for tr in traces:
        seed = _period_seed(tr.t, tr.v, tr.sync)
        p, _phi0, d = _fit_triangle_full(tr.t, tr.v, seed)
        periods.append(p)
        duties.append(d)
    period = float(np.median(periods))
    duty = float(np.median(duties))
    for tr in traces:
        segment(tr, config, period=period, duty=duty)
    return period, duty
