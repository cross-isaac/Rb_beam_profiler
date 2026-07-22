"""
Loading and segmenting raw oscilloscope traces.

Three on-disk formats are supported transparently:

1. Legacy export with an explicit time column:
       Time(s),CH2(V),CH4(V)
       -3.2844E-03,-1.156769E-01,4.061519E+00
       ...

2. Legacy "raw" export where the header encodes the time base and the file
   holds only samples (seen as `full0.csv` in the old data):
       CH2(V), t0 = -0.0040844s,  tInc = 5e-10s,
       -0.36...
   Here time is reconstructed as t = t0 + i*tInc.

3. Acquisition-runner "run" format (2026-07 onward, op-amp bypassed): a
   `#`-comment metadata header followed by four columns:
       # timestamp: ...
       # target_in: 0.20000
       time_s,CH2_V,CH3_V,CH4_V
   CH2 is the raw DC photodetector level (used to normalize to absolute
   absorption), CH3 the AC-coupled copy that resolves the peaks (fitted),
   CH4 the sync. The stage position comes from `target_in` (inches) and is
   converted to thou so all downstream geometry is unchanged.

Segmentation: the Toptica scan controller outputs a square-wave sync (recorded
on CH4) whose level flips at every sawtooth turning point, so each constant-
level stretch of the sync is exactly one ramp side (one up- or down-sweep
through the resonance). We split on those edges. If no sync is available,
a heuristic based on the smoothed signal derivative is used, or the user can
supply manual boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import AnalysisConfig


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RampSide:
    """One half-sweep (up or down) of the laser frequency ramp."""
    index: int                # 1-based side number within the trace
    t: np.ndarray             # time [s], bin-averaged
    v: np.ndarray             # signal [V], bin-averaged
    t_start: float            # untrimmed boundaries in the original trace
    t_end: float
    # DC transmission level over this side (run format only): mean of the raw
    # CH2 channel and its uncertainty (scatter + ADC quantization).
    dc_mean: float | None = None
    dc_err: float | None = None


@dataclass
class Trace:
    """A full scope acquisition for one probe position."""
    path: str
    t: np.ndarray             # full time vector [s] (possibly downsampled)
    v: np.ndarray             # signal [V]
    sync: np.ndarray | None   # sync channel [V], or None if absent
    sides: list = field(default_factory=list)   # list[RampSide], filled by segment()
    dc: np.ndarray | None = None    # raw DC photodetector channel (run format)
    meta: dict = field(default_factory=dict)    # '# key: value' header metadata
    position_thou: float | None = None          # stage position from metadata [thou]


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────
def _bin_average(t: np.ndarray, v: np.ndarray, n_bins: int):
    """Average (t, v) into n_bins equal-time bins.

    Scope exports oversample massively (up to 2e7 rows) and often repeat time
    stamps; bin-averaging both shrinks the arrays to a fittable size and
    reduces white noise without biasing the lineshape (bin width << linewidth).
    """
    if len(t) <= n_bins:
        return t, v
    edges = np.linspace(t[0], t[-1], n_bins + 1)
    idx = np.searchsorted(edges, t, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    t_out = np.bincount(idx, weights=t, minlength=n_bins)
    v_out = np.bincount(idx, weights=v, minlength=n_bins)
    good = counts > 0
    return t_out[good] / counts[good], v_out[good] / counts[good]


# Column roles in the run format (op-amp bypassed, raw + AC-coupled channels).
RUN_TIME_COL = "time_s"
RUN_DC_COL = "CH2_V"      # raw DC photodetector level
RUN_SIGNAL_COL = "CH3_V"  # AC-coupled signal — this is what gets fitted
RUN_SYNC_COL = "CH4_V"
IN_TO_THOU = 1000.0


def read_run_metadata(path: str) -> dict:
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


def load_trace(path: str, config: AnalysisConfig) -> Trace:
    """Load one scope CSV in any supported format.

    Run-format files are recognized by their `time_s` column (plus optional
    `#` metadata header). For legacy files an explicit time column always
    wins; the t0/tInc header form is only used when no time column exists
    (some scope exports carry a leftover "t0 = ..., tInc = ..." fragment in
    the header even WITH a time column).
    """
    # Skip any leading '#' metadata lines, then peek at the header line.
    meta = read_run_metadata(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        while header.startswith("#"):
            header = fh.readline()
    header_cols = [c.strip() for c in header.split(",")]

    dc = None
    position_thou = None
    if RUN_TIME_COL in header_cols and RUN_SIGNAL_COL in header_cols:
        # ── run format ────────────────────────────────────────────────────────
        df = pd.read_csv(path, comment="#")
        df.columns = [c.strip() for c in df.columns]
        t = df[RUN_TIME_COL].to_numpy(dtype=float)
        v = df[RUN_SIGNAL_COL].to_numpy(dtype=float)
        dc = df[RUN_DC_COL].to_numpy(dtype=float) if RUN_DC_COL in df.columns else None
        sync = df[RUN_SYNC_COL].to_numpy(dtype=float) if RUN_SYNC_COL in df.columns else None
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        if dc is not None:
            dc = dc[order]
        if sync is not None:
            sync = sync[order]
        if "target_in" in meta:
            try:
                position_thou = float(meta["target_in"]) * IN_TO_THOU
            except ValueError:
                pass
        return Trace(path=str(path), t=t, v=v, sync=sync, dc=dc,
                     meta=meta, position_thou=position_thou)

    if config.time_column in header_cols:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if config.signal_column not in df.columns:
            raise ValueError(
                f"{path}: expected signal column '{config.signal_column}', "
                f"found {list(df.columns)}"
            )
        t = df[config.time_column].to_numpy(dtype=float)
        v = df[config.signal_column].to_numpy(dtype=float)
        sync = (
            df[config.sync_column].to_numpy(dtype=float)
            if config.sync_column in df.columns
            else None
        )
        # Scope exports are usually time-ordered, but guard against repeats/reversals.
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        if sync is not None:
            sync = sync[order]
    elif "t0" in header and "tInc" in header:
        # Implicit time base in the header, samples in the signal column (or
        # the first column if the signal column name isn't present either).
        m_t0 = re.search(r"t0\s*=\s*([-+0-9.eE]+)", header)
        m_inc = re.search(r"tInc\s*=\s*([-+0-9.eE]+)", header)
        if not (m_t0 and m_inc):
            raise ValueError(f"Could not parse t0/tInc from header of {path}")
        t0, t_inc = float(m_t0.group(1)), float(m_inc.group(1))
        col = (header_cols.index(config.signal_column)
               if config.signal_column in header_cols else 0)
        v = pd.read_csv(path, usecols=[col]).iloc[:, 0].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        t = t0 + np.arange(len(v)) * t_inc
        sync = None
    else:
        raise ValueError(
            f"{path}: no '{config.time_column}' column and no t0/tInc header — "
            "unrecognized scope CSV format."
        )

    # Very large raw exports (up to 2e7 rows) are bin-averaged down right away:
    # segmentation and plotting need nowhere near that resolution, and each
    # side is re-binned later anyway.
    max_points = 400_000
    if len(t) > max_points:
        if sync is not None:
            _, sync = _bin_average(t, sync, max_points)
        t, v = _bin_average(t, v, max_points)

    return Trace(path=str(path), t=t, v=v, sync=sync)


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation
#
# The scan-controller sync (CH4) is a square wave at the sweep frequency, but
# in the recorded data its edges are NOT aligned with the sawtooth turning
# points — there is a fixed phase offset (about a quarter period in the 2025
# datasets). Using the raw sync edges as boundaries puts a ramp corner in the
# middle of each "side", which the polynomial background cannot follow, and
# the leftover corner then masquerades as a huge absorption peak.
#
# Strategy: take only the HALF-PERIOD from the sync (or from the signal's
# autocorrelation when no sync exists), then find the correct PHASE by scanning
# the boundary offset over one half-period and choosing the offset for which
# each segment is most monotonic/linear (a correctly-cut side is a clean ramp;
# a corner-straddling side is V-shaped and fits a line badly). The absorption
# dip adds the same misfit at every offset, so it does not bias the search.
# ──────────────────────────────────────────────────────────────────────────────
def _half_period_from_sync(t: np.ndarray, sync: np.ndarray) -> float | None:
    """Half-period of the sweep from the sync square wave (median edge spacing)."""
    lo, hi = np.percentile(sync, [5, 95])
    if hi - lo < 0.1:                      # sync channel is flat/absent
        return None
    state = sync > 0.5 * (lo + hi)
    edges = np.flatnonzero(np.diff(state.astype(int)) != 0)
    if len(edges) < 2:
        return None
    te = t[edges]
    # Debounce: glitchy syncs emit bursts of edges at (nearly) the same time,
    # which would halve the inferred period. Keep an edge only if it is at
    # least 20% of the largest spacing away from the previously kept edge.
    min_sep = 0.2 * float(np.max(np.diff(te))) if len(te) > 1 else 0.0
    kept = [float(te[0])]
    for x in te[1:]:
        if float(x) - kept[-1] >= min_sep:
            kept.append(float(x))
    te = np.asarray(kept)
    if len(te) < 2:
        return None
    if len(te) >= 3:
        # Lag-2 spacing = one full period, immune to a non-50% duty cycle
        # (the observed sync is high ~1.9 ms / low ~2.2 ms per half-sweep).
        return float(np.median(te[2:] - te[:-2])) / 2.0
    return float(te[1] - te[0])


def _half_period_auto(t: np.ndarray, v: np.ndarray) -> float:
    """Half-period estimate from the autocorrelation of the (smoothed) signal.

    The sawtooth dominates the trace, so the autocorrelation peaks at the full
    sweep period; the half-period is half of that.
    """
    # Downsample to keep the correlation cheap.
    n = min(len(v), 20000)
    idx = np.linspace(0, len(v) - 1, n).astype(int)
    vv = v[idx] - np.mean(v[idx])
    tt = t[idx]
    ac = np.correlate(vv, vv, mode="full")[n - 1:]
    ac /= ac[0]
    # First prominent maximum after the zero-lag peak decays.
    i = 1
    while i < n - 1 and ac[i] > 0:        # leave the zero-lag lobe
        i += 1
    if i >= n - 2:
        return (t[-1] - t[0]) / 4.0        # fallback: assume ~4 sides
    j = i + int(np.argmax(ac[i:]))
    period = float(tt[j] - tt[0])
    return period / 2.0


def _fold_average(t: np.ndarray, v: np.ndarray, period: float, n_bins: int = 400):
    """Average the trace folded by the sweep period -> one mean waveform w(phi)."""
    phi = np.mod(t - t[0], period)
    idx = np.clip((phi / period * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=v, minlength=n_bins)
    w = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    # Fill any empty bins by interpolation (rare, short traces only).
    if np.isnan(w).any():
        good = ~np.isnan(w)
        w = np.interp(np.arange(n_bins), np.flatnonzero(good), w[good], period=n_bins)
    return w


def _triangle_phase_fit(t: np.ndarray, v: np.ndarray, period: float):
    """Locate the sawtooth turning points by fitting a triangle wave.

    The trace is folded by the sweep period and the mean waveform is fit with
    an ideal (possibly asymmetric) triangle using a robust soft-L1 loss, so
    the absorption dip — however large — acts as a localized outlier and does
    not pull the corner phases. Returns (phi_min, phi_max): the phases (in
    seconds after t[0], modulo the period) of the two turning points.
    """
    from scipy.optimize import least_squares

    n_bins = 400
    w = _fold_average(t, v, period, n_bins)
    phi = (np.arange(n_bins) + 0.5) / n_bins            # fractional phase

    def triangle(params):
        phi0, duty, amp, off = params
        x = np.mod(phi - phi0, 1.0)                     # 0 at the minimum corner
        up = x < duty
        y = np.empty_like(x)
        y[up] = -0.5 + x[up] / duty                     # rising: -1/2 -> +1/2
        y[~up] = 0.5 - (x[~up] - duty) / (1.0 - duty)   # falling: +1/2 -> -1/2
        return amp * y + off

    phi0_init = float(np.argmin(w) + 0.5) / n_bins
    p0 = [phi0_init, 0.5, float(np.ptp(w)), float(np.mean(w))]
    res = least_squares(
        lambda p: triangle(p) - w, p0,
        bounds=([0.0, 0.2, 0.0, -np.inf], [1.0, 0.8, np.inf, np.inf]),
        loss="soft_l1", f_scale=max(float(np.ptp(w)) * 0.05, 1e-9),
    )
    phi0, duty = res.x[0] % 1.0, res.x[1]
    phi_min = phi0 * period
    phi_max = ((phi0 + duty) % 1.0) * period
    return phi_min, phi_max


def _boundaries(t: np.ndarray, v: np.ndarray, sync: np.ndarray | None) -> list[float]:
    """Ramp-side boundaries: period from sync/autocorrelation + triangle-fold phase.

    The recorded sync square wave has a fixed but unknown phase offset from the
    actual laser ramp, so only its PERIOD is used; the turning-point phases are
    measured from the signal itself via the robust triangle fit. Trace ends are
    included as boundaries so near-complete end segments are kept (segment()
    drops anything shorter than ~60% of a full side).
    """
    t_half = _half_period_from_sync(t, sync) if sync is not None else None
    if t_half is None:
        t_half = _half_period_auto(t, v)
    period = 2.0 * t_half

    phi_min, phi_max = _triangle_phase_fit(t, v, period)

    span0, span1 = float(t[0]), float(t[-1])
    inner: list[float] = []
    for phase in (phi_min, phi_max):
        b = span0 + phase
        while b < span1:
            if b > span0:
                inner.append(float(b))
            b += period
    inner.sort()
    return [span0] + inner + [span1]


def segment(trace: Trace, config: AnalysisConfig) -> Trace:
    """Split a trace into RampSides according to the configured method.

    Each side is edge-trimmed (turnaround artifacts) and bin-averaged down to
    `downsample_points_per_side`. Sides shorter than half the typical duration
    (incomplete sweeps at the trace ends) are dropped.
    """
    method = config.segmentation
    if method == "sync" and trace.sync is None:
        method = "auto"  # graceful fallback for files without a sync column

    if method == "manual" and config.manual_boundaries_s:
        bounds = sorted(float(b) for b in config.manual_boundaries_s)
    elif method == "sync":
        bounds = _boundaries(trace.t, trace.v, trace.sync)
    else:
        bounds = _boundaries(trace.t, trace.v, None)

    # Keep only sides of near-typical duration (drop partial sweeps at ends).
    durations = np.diff(bounds)
    typical = np.median(durations) if len(durations) else 0.0

    # ADC quantization step of the DC channel: the raw CH2 record is heavily
    # quantized (few distinct levels), so the LSB size — not the sample scatter
    # alone — limits how well the DC level is known.
    dc_lsb = 0.0
    if trace.dc is not None:
        levels = np.unique(trace.dc)
        if len(levels) > 1:
            dc_lsb = float(np.min(np.diff(levels)))

    sides: list[RampSide] = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        if (t1 - t0) < 0.6 * typical:
            continue
        trim = config.edge_trim_fraction * (t1 - t0)
        sel = (trace.t >= t0 + trim) & (trace.t <= t1 - trim)
        if sel.sum() < 50:
            continue
        ts, vs = _bin_average(trace.t[sel], trace.v[sel], config.downsample_points_per_side)
        dc_mean = dc_err = None
        if trace.dc is not None:
            dseg = trace.dc[sel]
            dc_mean = float(np.mean(dseg))
            # Scatter of the mean, floored at half an LSB (quantization limit).
            dc_err = max(float(np.std(dseg) / np.sqrt(len(dseg))), 0.5 * dc_lsb)
        sides.append(RampSide(index=len(sides) + 1, t=ts, v=vs, t_start=t0, t_end=t1,
                              dc_mean=dc_mean, dc_err=dc_err))

    trace.sides = sides
    return trace
