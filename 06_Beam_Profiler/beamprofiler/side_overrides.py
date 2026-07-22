"""
Manual per-side accept/reject verdicts.

The automatic tests (SNR, dAIC, linewidth runaway) are deliberately blunt: they
have to work unattended across a whole run. Looking at the `02_side_fits` images
afterwards, some verdicts are obviously wrong in either direction — a clean line
just under `snr_min` at a beam-edge position, or a "fit" that clearly latched
onto a ramp wiggle while passing every threshold. This module lets those two
cases be corrected by hand from the GUI image browser without loosening a
threshold run-wide (which would change every other side too).

The verdicts live in `manual_overrides.json` in the OUTPUT folder, keyed by the
side's stable label, `<file stem>:s<side index>` — the same key that appears in
`03_profile/side_details.csv`:

    {
      "version": 1,
      "note": "...",
      "overrides": {"point_0002:s1": "accept", "point_0008:s4": "reject"}
    }

Overriding a fit is a scientific claim about the data, so nothing here is
silent. An overridden side keeps BOTH verdicts (`auto_accepted` from the tests,
`accepted` after the override) and they both go into `side_details.csv`; the
side image says MANUAL OVERRIDE in its verdict box; the profile and summary
figures carry a footnote counting the overrides; `results.json` records the full
override map; and the pipeline logs every one it applies.

Scope: overrides are applied to the FINAL fitting pass only, never to the pass
that measures the reference linewidth. A side kept by hand should not be able to
redefine the matched-filter template for the whole run.
"""

from __future__ import annotations

import json
from pathlib import Path

OVERRIDES_FILENAME = "manual_overrides.json"
VALID_VERDICTS = ("accept", "reject")

_NOTE = ("Manual per-side verdicts applied on top of the automatic SNR / dAIC / "
         "linewidth tests. Key = '<file stem>:s<side>' as in side_details.csv; "
         "value = 'accept' or 'reject'. Delete an entry to fall back to the "
         "automatic verdict. Re-run the pipeline for changes to take effect.")


def overrides_path(output_folder: str | Path) -> Path:
    return Path(output_folder) / OVERRIDES_FILENAME


def load_overrides(output_folder: str | Path) -> dict[str, str]:
    """Read the manual verdicts for an output folder ({} if there are none).

    Unknown keys and malformed verdicts are dropped rather than raised on: a
    stale override left over from a previous data folder must not stop a run.
    """
    path = overrides_path(output_folder)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    raw = data.get("overrides", {}) if isinstance(data, dict) else {}
    return {str(k): str(v).lower() for k, v in raw.items()
            if str(v).lower() in VALID_VERDICTS}


def save_overrides(output_folder: str | Path, overrides: dict[str, str]) -> Path:
    """Write the manual verdicts back, dropping any empty/invalid entries."""
    clean = {k: v.lower() for k, v in sorted(overrides.items())
             if v and v.lower() in VALID_VERDICTS}
    path = overrides_path(output_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "note": _NOTE, "overrides": clean},
                               indent=2), encoding="utf-8")
    return path


def apply_override(peak, verdict: str | None) -> bool:
    """Force one PeakFit's verdict. Returns True if the effective verdict moved.

    The automatic result is preserved in `auto_accepted` and in the reject
    reasons, so the override is always auditable from the CSV or the figure.
    """
    if verdict is None or verdict.lower() not in VALID_VERDICTS:
        return False
    verdict = verdict.lower()
    peak.manual_override = verdict
    was = peak.accepted
    peak.accepted = (verdict == "accept")
    if verdict == "accept":
        # Keep the automatic complaints visible — the point is to see what was
        # overruled, not to erase it.
        peak.reject_reasons = [f"overruled by hand: {r}" for r in peak.reject_reasons]
    else:
        peak.reject_reasons = list(peak.reject_reasons) + ["rejected by hand"]
    return peak.accepted != was
