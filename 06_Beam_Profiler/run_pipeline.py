"""
Command-line entry point for the beam-profiler pipeline (run-format data).

Examples
--------
    python run_pipeline.py --data run_20260720_1642 --out results\\run_072026 --label Rb_072026

    # with the dopant-density estimate:
    python run_pipeline.py --data <folder> --out <results> --ar-flow 0.01

Any setting not exposed as a flag can be changed by editing a config JSON:
    python run_pipeline.py --data <folder> --out <results> --config my_settings.json
(create a template with:  python run_pipeline.py --write-default-config my_settings.json)
"""

import argparse
import sys

from beamprofiler import AnalysisConfig, run_full_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rb beam characterization: run-format scope CSVs -> divergence/flux.")
    ap.add_argument("--data", help="folder of run-format position CSVs (point_*.csv)")
    ap.add_argument("--out", help="output folder for results and images")
    ap.add_argument("--config", help="JSON config file (optional; defaults otherwise)")
    ap.add_argument("--label", default="", help="dataset label used in plot titles")
    ap.add_argument("--corner-fraction", type=float,
                    help="width of each corner anchor cluster (fraction of the side)")
    ap.add_argument("--snr-min", type=float, help="reject side fits below this SNR")
    ap.add_argument("--runaway-fraction", type=float, metavar="F",
                    help="'linewidth ran away' sensitivity: reject a side whose fitted "
                         "HWHM reaches F x the upper linewidth bound (default 0.97; "
                         "1.0 = only exactly-pinned fits; 0 = test off)")
    ap.add_argument("--gamma-max", type=float, metavar="R",
                    help="upper linewidth bound as a multiple of the reference HWHM "
                         "(default 2.5) — sets where 'ran away' starts")
    ap.add_argument("--verdict-mode", choices=["auto", "flag"],
                    help="'auto' (default) rejects failing sides; 'flag' accepts every "
                         "side and marks the failing ones suspect for manual review "
                         "(changes the profile — noise edges enter until you reject them)")
    ap.add_argument("--no-overrides", action="store_true",
                    help="ignore manual_overrides.json in the output folder "
                         "(reproduce a purely automatic analysis)")
    ap.add_argument("--rejected-mode", choices=["zero", "exclude", "keep"],
                    help="how a failed side counts in the per-position average")
    ap.add_argument("--error-mode", choices=["STD", "SEM", "upper_bound"],
                    help="per-position error bar")
    ap.add_argument("--z-probe", type=float, help="nozzle->probe distance [in]")
    ap.add_argument("--oven-temp", type=float, help="nozzle temperature [C]")
    ap.add_argument("--ar-flow", type=float, help="Ar flow [SCCM] (enables dopant density)")
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
    if args.corner_fraction is not None:
        config.corner_cluster_fraction = args.corner_fraction
    if args.snr_min is not None:
        config.snr_min = args.snr_min
    if args.runaway_fraction is not None:
        config.linewidth_runaway_fraction = args.runaway_fraction
    if args.gamma_max is not None:
        config.gamma_rel_bounds = (config.gamma_rel_bounds[0], args.gamma_max)
    if args.verdict_mode is not None:
        config.side_verdict_mode = args.verdict_mode
    if args.no_overrides:
        config.use_manual_overrides = False
    if args.rejected_mode is not None:
        config.rejected_mode = args.rejected_mode
    if args.error_mode is not None:
        config.error_mode = args.error_mode
    if args.z_probe is not None:
        config.z_probe_in = args.z_probe
    if args.oven_temp is not None:
        config.oven_temp_C = args.oven_temp
    if args.ar_flow is not None:
        config.ar_flow_sccm = args.ar_flow

    run_full_pipeline(data_folder=args.data, output_folder=args.out, config=config,
                      progress=print, label=args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
