"""Train and evaluate the RCDIFO comparison."""

from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rcdifo.yaml")
    parser.add_argument(
        "--stage", choices=["train", "evaluate", "summarize", "all"], default="all"
    )
    parser.add_argument("--main-output", help="Matching benchmark output directory")
    parser.add_argument("--output")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    from experiments.rcdifo.engine import Runner, summarize

    if args.stage == "summarize":
        import yaml

        config = yaml.safe_load(Path(args.config).read_text())
        summarize(args.output or config["project"]["output_dir"])
        return
    with Runner(
        args.config,
        Path(__file__).resolve().parent,
        device=args.device,
        datasets=args.datasets,
        seeds=args.seeds,
        data_root=args.data_root,
        output=args.output,
        main_output=args.main_output,
    ) as runner:
        if args.stage in ["train", "all"]:
            runner.train()
        if args.stage in ["evaluate", "all"]:
            runner.evaluate()
        runner.summarize()


if __name__ == "__main__":
    main()
