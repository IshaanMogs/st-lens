"""Run the ST-LENS fast-track experiment (synthetic data) and write artefacts + report.

python scripts/run_experiment.py            # validation only
python scripts/run_experiment.py --final    # also evaluates test days (once!)
python scripts/run_experiment.py --quick    # tiny config for a smoke test
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
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--report", type=Path, default=Path("docs/results.md"))
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
    run(cfg, args.out, final=args.final, log=lambda m: print(m, flush=True))
    if not args.quick:
        write(args.out / "results.json", args.report)
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
