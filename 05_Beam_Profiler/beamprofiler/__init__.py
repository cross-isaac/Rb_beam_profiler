"""
beamprofiler — automated Rb atomic-beam characterization from absorption spectroscopy.

This package turns raw oscilloscope traces (one CSV per probe-beam position) into
final beam-characterization results:

    raw scope traces
        └─> ramp segmentation (trace_io)
        └─> polynomial background removal (ramp_removal)
        └─> Lorentzian absorption-peak fits + noise rejection (peak_fitting)
        └─> spatial profile A(Δy) + Gaussian/Lorentzian/pseudo-Voigt fits,
            divergence, capture efficiency (profile_analysis)
        └─> Beer-Lambert flux + dopant density (flux)
        └─> summary figures & machine-readable results (pipeline, plotting)

Physics background: Isaac Cross, Senior Thesis 2026, Chapters 4-5
(85Rb D2 F=3 -> F'=4 absorption spectroscopy of an effusive Rb(-Ar) beam).
"""

__version__ = "1.0.0"

from .config import AnalysisConfig
from .pipeline import run_full_pipeline
