#!/usr/bin/env python3
"""Preflight checks for AP4FED experiment runners."""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


COMMON_PYTHON_MODULES: Sequence[tuple[str, str]] = (
    ("flwr", "flwr"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("psutil", "psutil"),
    ("yaml", "yaml"),
    ("scipy", "scipy"),
    ("Pillow", "PIL"),
    ("torchgan", "torchgan"),
    ("scikit-learn", "sklearn"),
    ("scikit-image", "skimage"),
    ("grad-cam", "pytorch_grad_cam"),
    ("rich", "rich"),
)


DOCKER_PYTHON_MODULES: Sequence[tuple[str, str]] = (
    ("docker-sdk", "docker"),
)


COMMON_REPO_FILES: Sequence[Path] = (
    ROOT / "build_paper_experiments.py",
    ROOT / "build_paper_experiments_500clients_k5.py",
    ROOT / "Local" / "pyproject.toml",
)


DOCKER_REPO_FILES: Sequence[Path] = (
    ROOT / "Docker" / "Dockerfile.server",
    ROOT / "Docker" / "Dockerfile.client",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify AP4FED experiment environment")
    parser.add_argument("--mode", choices=("local", "docker", "both"), default="local")
    parser.add_argument("--ollama-url", help="Optional Ollama-compatible endpoint to probe")
    return parser.parse_args()


def run_command(cmd: Sequence[str]) -> CheckResult:
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return CheckResult(" ".join(cmd), False, "command not found")

    if completed.returncode != 0:
        detail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else f"exit code {completed.returncode}"
        return CheckResult(" ".join(cmd), False, detail)

    output = completed.stdout.strip().splitlines()
    detail = output[0] if output else "ok"
    return CheckResult(" ".join(cmd), True, detail)


def check_python_module(label: str, import_name: str) -> CheckResult:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(label, False, f"{type(exc).__name__}: {exc}")

    version = getattr(module, "__version__", None)
    return CheckResult(label, True, version or "import ok")


def check_repo_file(path: Path) -> CheckResult:
    if path.exists():
        return CheckResult(str(path.relative_to(ROOT)), True, "present")
    return CheckResult(str(path.relative_to(ROOT)), False, "missing")


def check_command_available(name: str) -> CheckResult:
    found = shutil.which(name)
    if not found:
        return CheckResult(name, False, "not on PATH")
    return CheckResult(name, True, found)


def probe_ollama(base_url: str) -> CheckResult:
    candidates = [base_url.rstrip("/"), f"{base_url.rstrip('/')}/api/tags"]
    for candidate in candidates:
        try:
            with urllib.request.urlopen(candidate, timeout=5) as response:
                return CheckResult("ollama-endpoint", True, f"{candidate} -> HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            return CheckResult("ollama-endpoint", True, f"{candidate} -> HTTP {exc.code}")
        except Exception:
            continue
    return CheckResult("ollama-endpoint", False, f"unreachable: {base_url}")


def print_results(title: str, results: Iterable[CheckResult]) -> int:
    failures = 0
    print(f"\n[{title}]")
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"- {status:<4} {result.name}: {result.detail}")
        if not result.ok:
            failures += 1
    return failures


def main() -> int:
    args = parse_args()
    failures = 0

    module_checks: List[CheckResult] = [check_python_module(label, import_name) for label, import_name in COMMON_PYTHON_MODULES]
    repo_checks: List[CheckResult] = [check_repo_file(path) for path in COMMON_REPO_FILES]

    if args.mode in {"docker", "both"}:
        module_checks.extend(check_python_module(label, import_name) for label, import_name in DOCKER_PYTHON_MODULES)
        repo_checks.extend(check_repo_file(path) for path in DOCKER_REPO_FILES)

    failures += print_results(
        "python-modules",
        module_checks,
    )
    failures += print_results("repo-files", repo_checks)
    failures += print_results(
        "base-commands",
        [
            check_command_available("flower-simulation"),
            run_command([sys.executable, "--version"]),
            run_command([sys.executable, "-m", "pip", "--version"]),
        ],
    )

    if args.mode in {"docker", "both"}:
        failures += print_results(
            "docker-runtime",
            [
                check_command_available("docker"),
                run_command(["docker", "compose", "version"]),
                run_command(["docker", "info"]),
            ],
        )

    if args.ollama_url:
        failures += print_results("llm-endpoint", [probe_ollama(args.ollama_url)])

    if failures:
        print(f"\nEnvironment check failed with {failures} issue(s).", file=sys.stderr)
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
