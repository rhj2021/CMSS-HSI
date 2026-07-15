from __future__ import annotations
import argparse
from cmss_grsl.runner import CMSSRunner


def main() -> None:
    ap = argparse.ArgumentParser(description='CMSS GRSL experiment runner')
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage', default='all', choices=['inspect', 'prepare', 'train', 'evaluate', 'summarize', 'all'])
    ap.add_argument('--device', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    runner = CMSSRunner(args.config, device_override=args.device, force=args.force)
    if args.stage in ('inspect', 'all'):
        runner.inspect()
    if args.stage in ('prepare', 'all'):
        runner.prepare()
    if args.stage in ('train', 'all'):
        runner.train()
    if args.stage in ('evaluate', 'all'):
        runner.evaluate()
    if args.stage in ('summarize', 'all'):
        bundle = runner.summarize()
        print(bundle)


if __name__ == '__main__':
    main()
