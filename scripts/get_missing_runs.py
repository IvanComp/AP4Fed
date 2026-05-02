#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        "build_paper_experiments.py",
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--repeat",
        str(args.repeat),
        "--rounds",
        str(args.rounds),
        "--dry-run",
        "--quiet",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip() or proc.stdout.strip(), file=sys.stderr)
        return proc.returncode

    payload = json.loads(proc.stdout)
    print(int(payload.get("scheduled_new_runs", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
