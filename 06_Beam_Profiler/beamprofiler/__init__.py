"""
beamprofiler — automated Rb atomic-beam characterization from raw scope traces.

Simplified rebuild (v2): reads the acquisition-runner run format only, uses one
peak-aware corners background, and locates each side's absorption peak with an
independent matched filter (robust to the laser drift that wanders the line
across the sweep). See the README for the full pipeline description.
"""

from .config import AnalysisConfig
from .pipeline import run_full_pipeline, run_extraction, run_profile

__all__ = ["AnalysisConfig", "run_full_pipeline", "run_extraction", "run_profile"]
