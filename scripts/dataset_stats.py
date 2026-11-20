"""Show the local dataset's label/session coverage without training a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.dataset.stats import dataset_stats
from backend.ml.train import DatasetError


def _print_table(stats: dict[str, Any]) -> None:
    print(f"Dataset: {stats['path']}")
    print(
        f"Linhas: {stats['usable_rows']} utilizáveis / {stats['total_rows']} totais "
        f"({stats['total_rows'] - stats['usable_rows']} descartadas)"
    )
    print("\nRótulo        Linhas  Sessões  Segundos")
    for label, values in stats["labels"].items():
        print(
            f"{label:<14}{values['rows']:>6}  {values['sessions']:>7}  "
            f"{values['seconds']:>8.1f}"
        )

    if stats["dropped"]:
        reasons = ", ".join(f"{reason}={count}" for reason, count in stats["dropped"].items())
        print(f"\nDescartadas: {reasons}")
    if stats["problems"]:
        print("\nProblemas:")
        for problem in stats["problems"]:
            print(f"  - {problem}")
    print(f"\nCobertura por sessões pronta: {'sim' if stats['ready'] else 'não'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect dataset coverage by label and recording session."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset CSV path (default: data/training/dataset.csv)",
    )
    parser.add_argument("--json", action="store_true", help="print the endpoint-shaped JSON")
    args = parser.parse_args()

    try:
        stats = dataset_stats() if args.dataset is None else dataset_stats(args.dataset)
    except (OSError, DatasetError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        _print_table(stats)
    return 0 if stats["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
