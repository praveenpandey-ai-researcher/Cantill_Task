"""Command-line entry point.

    python -m spendpipe run --config configs/company_A.yaml --out outputs/A.csv
    python -m spendpipe profile <file.csv>          (drafts a config; later stage)
"""
from __future__ import annotations

import argparse
import logging
import sys

from spendpipe.config import load_config

log = logging.getLogger("spendpipe")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    from spendpipe.pipeline import run_to_file   # heavy imports only when needed
    cfg = load_config(args.config, data_dir=args.data_dir)
    log.info("Loaded config for company %s (%d source files)", cfg.company, len(cfg.sources))
    run_to_file(cfg, args.out)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    log.warning("profile command not implemented yet.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spendpipe", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline for one company")
    p_run.add_argument("--config", required=True, help="path to company YAML config")
    p_run.add_argument("--out", required=True, help="output CSV path")
    p_run.add_argument("--data-dir", default=None,
                       help="override the config's data_dir (e.g. for blind-test files)")
    p_run.set_defaults(func=cmd_run)

    p_prof = sub.add_parser("profile", help="scan a file and draft a config")
    p_prof.add_argument("file")
    p_prof.set_defaults(func=cmd_profile)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
