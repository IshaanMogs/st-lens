"""Run the ST-LENS fast-track experiment (synthetic data) and write artefacts + report.

python scripts/run_experiment.py                  # validation only -> artifacts/
python scripts/run_experiment.py --final          # + one-time test evaluation -> artifacts/,
                                                  #   report -> docs/results.md
python scripts/run_experiment.py --quick --final  # tiny smoke run -> demo_quick/ (report inside)

A quick run never writes to artifacts/ or docs/ unless --out/--report say so explicitly.
"""

import argparse
from pathlib import Path

from stlens.datasets.build import DataConfig
from stlens.experiment import ExperimentConfig, run
from stlens.report import write
from stlens.training.trainer import TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true", help="evaluate the test days (logged)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: artifacts/, or demo_quick/ with --quick)",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="results report (default: docs/results.md, or <out>/results.md with --quick)",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    if args.quick:
        cfg = ExperimentConfig(
            data=DataConfig(n_days=4, n_val_days=1, n_test_days=1, sim_steps_per_day=6_000),
            train=TrainConfig(epochs=2),
            seeds=(0,),
        )
    else:
        cfg = ExperimentConfig(train=TrainConfig(epochs=8, patience=3), seeds=tuple(args.seeds))
    out = args.out or Path("demo_quick" if args.quick else "artifacts")
    report = args.report or (out / "results.md" if args.quick else Path("docs/results.md"))
    run(cfg, out, final=args.final, log=lambda m: print(m, flush=True))
    write(out / "results.json", report)
    print(f"artefacts written to {out}/, report written to {report}")


if __name__ == "__main__":
    main()
