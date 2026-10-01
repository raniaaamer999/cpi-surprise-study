"""Runs the full CPI surprise study from one command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]


def _python_module(module: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *arguments)


def _python_script(path: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, path, *arguments)


def _required_inputs() -> tuple[Path, ...]:
    return (
        ROOT / "data/private/cpi_bloomberg_private.xlsx",
        ROOT / "data/external/daily_market_data.csv",
    )


def _check_inputs(*, refresh_public_data: bool) -> None:
    # Checks that the files needed for an offline run are available.
    required = list(_required_inputs())
    if refresh_public_data:
        required = [path for path in required if "data/external" not in str(path)]
    missing = [path.relative_to(ROOT) for path in required if not path.is_file()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Missing required input files:\n{rendered}")


def _build_steps(args: argparse.Namespace) -> list[Step]:
    # Builds the commands in the order needed to recreate the study.
    steps: list[Step] = []
    if args.refresh_public_data:
        steps.append(
            Step(
                "Download public daily market data",
                _python_module(
                    "macro_surprise.data.market_client",
                    "--start",
                    args.start,
                    "--end",
                    args.end,
                ),
            )
        )

    steps.extend(
        [
            Step(
                "Clean original Bloomberg CPI releases",
                _python_module("macro_surprise.data.bloomberg_cpi"),
            ),
            Step(
                "Build CPI surprise and market reaction dataset",
                _python_module("macro_surprise.analysis.build_event_dataset"),
            ),
            Step(
                "Estimate CPI regressions and statistical tests",
                _python_module("macro_surprise.analysis.statistical_analysis"),
            ),
            Step(
                "Generate six research charts",
                _python_script("scripts/build_charts.py"),
            ),
        ]
    )
    if not args.skip_tests:
        steps.append(
            Step("Run the automated test suite", (sys.executable, "-m", "pytest"))
        )
    return steps


def _run_step(step: Step, number: int, total: int, *, dry_run: bool) -> None:
    # Adds the source folder to Python before running each command.
    print(f"\n[{number}/{total}] {step.name}", flush=True)
    print("  $ " + " ".join(step.command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    result = subprocess.run(
        step.command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"\nPipeline stopped at step {number}: {step.name}\n"
            "Fix the error shown above, then run this command again."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the complete CPI research project in order."
    )
    parser.add_argument(
        "--refresh-public-data",
        action="store_true",
        help="Redownload FRED/Yahoo market data (uses internet).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest at the end (not recommended for a final validation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without changing any files.",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()
    for label in ("start", "end"):
        try:
            date.fromisoformat(getattr(args, label))
        except ValueError:
            parser.error(f"--{label} must be YYYY-MM-DD")
    return args


def main() -> None:
    args = _parse_args()
    _check_inputs(refresh_public_data=args.refresh_public_data)
    steps = _build_steps(args)
    print("US CPI Surprise and Cross Asset Reaction Study")
    print(f"Project: {ROOT}")
    print(f"Mode: {'preview only' if args.dry_run else 'rebuild'}")
    for number, step in enumerate(steps, start=1):
        _run_step(step, number, len(steps), dry_run=args.dry_run)
    if args.dry_run:
        print("\nDry run complete. No pipeline commands were executed.")
    elif args.skip_tests:
        print("\nSUCCESS: analysis and figures completed (tests skipped).")
    else:
        print("\nSUCCESS: analysis, figures, and checks completed.")


if __name__ == "__main__":
    main()
