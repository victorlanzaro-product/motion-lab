"""Fail the CI gate when the manifest SHA differs from the checked-out SHA."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def verify_sha(expected: str, actual: str) -> str:
    """Return the tested SHA or reject missing, malformed, or mismatched values."""
    if not SHA_RE.fullmatch(expected) or not SHA_RE.fullmatch(actual):
        raise ValueError("expected and actual SHA must be full 40-character lowercase hex")
    if expected != actual:
        raise ValueError(f"tested SHA mismatch: expected {expected}, checked out {actual}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        sha = verify_sha(args.expected, args.actual)
    except ValueError as exc:
        parser.error(str(exc))
    args.manifest.write_text(f"tested_sha={sha}\n", encoding="utf-8")
    print(f"tested_sha={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
