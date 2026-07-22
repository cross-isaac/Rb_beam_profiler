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

Segmentation: the ramp sides are set by the CH4 sync alone. The sync is a square
wave at the sweep frequency, but its edges are NOT the sawtooth turning points —
there is a fixed quarter-period offset: the laser reaches its extreme half-way
through each sync level, so the turning points are the MIDPOINTS of the CH4 level
windows and each ramp side runs midpoint-to-midpoint, straddling a CH4 edge.
Each resulting side is one up- or down-sweep through resonance.
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
# Segmentation — the ramp sides come from the CH4 sync alone
#
# The scan controller's sync (CH4) is a square wave at the sweep frequency, but
# its edges are NOT the sawtooth turning points: there is a fixed quarter-period
# offset (documented in 05_Beam_Profiler and re-measured on the 2026-07-20 and
# 2026-07-22 runs). The laser reaches its extreme half-way through each sync
# level, so the turning points are the MIDPOINTS of the CH4 level windows and
# each ramp side runs midpoint-to-midpoint, straddling a CH4 edge.
#
# Taking the boundaries from CH4 alone matters because the sync is a physical
# feature of the scan drive: it is identical in every file of a run and entirely
# unaffected by the absorption. Fitting the sawtooth to the CH3 signal instead
# fails exactly when the line is strong or broad enough to warp the waveform —
# on the 2026-07-22 run a CH3 triangle fit scattered the period over
# 3.92-4.14 ms and the turning points by more than 1 ms between files whose CH4
# edges were identical to the sample.
# ──────────────────────────────────────────────────────────────────────────────
def _sync_edges(t: np.ndarray, sync: np.ndarray | None) -> np.ndarray:
    """Times [s] of the CH4 level transitions, debounced."""
    if sync is None:
        return np.array([])
    lo, hi = np.percentile(sync, [5, 95])
    if hi - lo < 0.1:                      # sync channel flat/absent
        return np.array([])
    state = sync > 0.5 * (lo + hi)
    idx = np.flatnonzero(np.diff(state.astype(int)) != 0)
    if len(idx) == 0:
        return np.array([])
    te = t[idx]
    # Debounce glitchy syncs: a burst of transitions at (nearly) the same time
    # would otherwise halve the inferred period. Keep an edge only if it is well
    # separated from the previously kept one.
    min_sep = 0.2 * float(np.max(np.diff(te))) if len(te) > 1 else 0.0
    kept = [float(te[0])]
    for x in te[1:]:
        if float(x) - kept[-1] >= min_sep:
            kept.append(float(x))
    return np.asarray(kept)


def _turning_points(t: np.ndarray, sync: np.ndarray) -> tuple[list[float], float]:
    """Sawtooth turning points [s] and the half-period, from CH4 alone.

    The turning points are the MIDPOINTS of the complete CH4 level windows — the
    quarter-period offset from the sync edges. The first and last windows are cut
    off by the acquisition window, so their midpoints are not real turning points;
    the periodic grid is extrapolated from the complete ones to cover the whole
    record instead.
    """
    edges = _sync_edges(t, sync)
    if len(edges) < 2:
        raise ValueError(
            "CH4 sync has fewer than two usable edges — cannot segment this "
            "trace. Check that CH4 was recorded and is not flat.")
    mids = 0.5 * (edges[:-1] + edges[1:])          # complete windows only
    half_period = (float(np.median(np.diff(mids))) if len(mids) >= 2
                   else float(np.median(np.diff(edges))))
    if not np.isfinite(half_period) or half_period <= 0:
        raise ValueError("Could not determine the sweep period from CH4.")

    span0, span1 = float(t[0]), float(t[-1])
    first = float(mids[0])
    k_lo = int(np.floor((span0 - first) / half_period)) - 1
    k_hi = int(np.ceil((span1 - first) / half_period)) + 1
    grid = [first + k * half_period for k in range(k_lo, k_hi + 1)]
    return [x for x in grid if span0 < x < span1], half_period


def _build_sides(trace: Trace, bounds, expected_dur, config) -> Trace:
    """Turn boundaries into RampSides (edge-trim, bin-average, drop partial ends).

    Every full ramp side lasts one half-period, so the partial ramps at the two
    ends of the record are the ones shorter than `min_side_fraction` of that.
    """
    dc_lsb = 0.0
    if trace.dc is not None:
        levels = np.unique(trace.dc)
        if len(levels) > 1:
            dc_lsb = float(np.min(np.diff(levels)))

    sides: list[RampSide] = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        dur = t1 - t0
        if dur < config.min_side_fraction * expected_dur:   # truncated end ramp
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


def segment(trace: Trace, config: AnalysisConfig) -> Trace:
    """Split one trace into RampSides using its CH4 sync."""
    turns, half_period = _turning_points(trace.t, trace.sync)
    bounds = [float(trace.t[0])] + turns + [float(trace.t[-1])]
    return _build_sides(trace, bounds, half_period, config)


def segment_all(traces: list[Trace], config: AnalysisConfig) -> tuple[float, float]:
    """Segment every trace of a run from its own CH4 sync.

    The sync is a fixed feature of the scan drive, so this is already consistent
    across the run — no shared geometry has to be imposed. Returns the median
    (sweep period, duty) for logging; the duty is 0.5 by construction, since
    every ramp side is one half-period between consecutive turning points.
    """
    halves = []
    for tr in traces:
        segment(tr, config)
        _turns, half_period = _turning_points(tr.t, tr.sync)
        halves.append(half_period)
    return 2.0 * float(np.median(halves)), 0.5
