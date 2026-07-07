"""
Command-line entry point for the beam-profiler pipeline.

Examples
--------
Basic run (divergence only):
    python run_pipeline.py --data ..\\04_Beam_Characterization\\absorption_data\\data_112825 --out results\\rb_ar_zone1

With absolute flux and dopant density:
    python run_pipeline.py --data <data folder> --out <results folder> ^
        --flux <flux trace file or folder> --ar-flow 0.01

Two-zone (Method B):
    python run_pipeline.py --data <zone1 folder> --zone2 <zone2 folder> --out <results>

Any setting not exposed as a flag can be changed by editing a config JSON:
    python run_pipeline.py --data <folder> --out <results> --config my_settings.json
(create a template with:  python run_pipeline.py --write-default-config my_settings.json)
"""

import argparse
import sys

from beamprofiler import AnalysisConfig, run_full_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rb beam characterization: raw scope CSVs -> divergence/flux results.")
    ap.add_argument("--data", help="folder of position CSVs (e.g. 450.csv) for Zone 1")
    ap.add_argument("--out", help="output folder for results and images")
    ap.add_argument("--flux", help="unattenuated flux trace CSV or folder (optional)")
    ap.add_argument("--zone2", help="second-zone data folder for Method B (optional)")
    ap.add_argument("--config", help="JSON config file (optional; defaults otherwise)")
    ap.add_argument("--label", default="", help="dataset label used in plot titles")
    # A few common settings as direct flags:
    ap.add_argument("--ramp-degree", type=int, choices=[1, 2],
                    help="background polynomial degree (default 2)")
    ap.add_argument("--z-probe", type=float,
                    help="nozzle->probe distance in inches (2.46 near, 12.3 far)")
    ap.add_argument("--oven-temp", type=float, help="nozzle temperature in C")
    ap.add_argument("--ar-flow", type=float,
                    help="Ar flow in SCCM (enables dopant-density estimate)")
    ap.add_argument("--flux-position", type=float,
                    help="stage position of the flux reference trace")
    ap.add_argument("--write-default-config", metavar="PATH",
                    help="write a default config JSON to PATH and exit")
    args = ap.parse_args()

    if args.write_default_config:
        AnalysisConfig().save(args.write_default_config)
        print(f"Wrote default config to {args.write_default_config}")
        return 0

    if not args.data or not args.out:
        ap.error("--data and --out are required (or use --write-default-config)")

    config = AnalysisConfig.load(args.config) if args.config else AnalysisConfig()
    if args.ramp_degree is not None:
        config.ramp_degree = args.ramp_degree
    if args.z_probe is not None:
        config.z_probe_in = args.z_probe
    if args.oven_temp is not None:
        config.oven_temp_C = args.oven_temp
    if args.ar_flow is not None:
        config.ar_flow_sccm = args.ar_flow
    if args.flux_position is not None:
        config.flux_reference_position = args.flux_position

    run_full_pipeline(
        data_folder=args.data,
        output_folder=args.out,
        config=config,
        flux_source=args.flux,
        zone2_folder=args.zone2,
        progress=print,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
