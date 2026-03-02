#!/usr/bin/env python3
import argparse
import copy
import itertools
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_METHODS = [
    "GradCAM",
    "HiResCAM",
    "ScoreCAM",
    "GradCAMPlusPlus",
    "AblationCAM",
    "XGradCAM",
    "EigenCAM",
    "FullGrad",
]

DEFAULT_MODELS = ["CNN 16k", "shufflenet_v2_x0_5"]
DEFAULT_DATASETS = ["CIFAR-10", "FashionMNIST"]
DEFAULT_CRITERIA = ["Min", "Max"]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def parse_csv_arg(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def detect_cuda_gpu_count() -> int:
    """Match GUI launcher behavior: detect CUDA GPUs via nvidia-smi, fallback to torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("GPU ")]
            if lines:
                return len(lines)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass

    return 0


def infer_unique_values(cfg: Dict[str, Any], key: str) -> List[str]:
    vals = []
    for cd in cfg.get("client_details", []):
        val = str(cd.get(key, "")).strip()
        if val and val not in vals:
            vals.append(val)
    return vals


def sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")


def experiment_label(dataset: str, model: str, method: str, criteria: str, repeat_idx: int) -> str:
    safe = lambda s: "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
    return f"{safe(dataset)}__{safe(model)}__{safe(method)}__{safe(criteria)}__r{repeat_idx:02d}"


def apply_experiment_to_config(
    base_cfg: Dict[str, Any],
    dataset: str,
    model: str,
    method: str,
    criteria: str,
    rounds_override: Optional[int],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    if rounds_override is not None:
        cfg["rounds"] = int(rounds_override)

    selector = cfg.setdefault("patterns", {}).setdefault("client_selector", {})
    selector["enabled"] = True
    params = selector.setdefault("params", {})
    params["selection_strategy"] = "SSIM-Based"
    params["selection_criteria"] = criteria
    params["explainer_type"] = method

    # Force 5 clients with requested data distributions:
    # Client 1-2-3 IID, Client 4-5 non-IID.
    cfg["clients"] = 5
    existing = cfg.get("client_details", [])
    template = existing[0] if existing else {}
    new_details = []
    for cid in range(1, 6):
        cd = copy.deepcopy(template)
        cd["client_id"] = cid
        cd["cpu"] = int(cd.get("cpu", 5))
        cd["ram"] = int(cd.get("ram", 2))
        cd["dataset"] = dataset
        cd["model"] = model
        cd["epochs"] = int(cd.get("epochs", 1))
        cd["delay_combobox"] = cd.get("delay_combobox", "No")
        cd["data_persistence_type"] = cd.get("data_persistence_type", "Same Data")
        cd["data_distribution_type"] = "IID" if cid <= 3 else "non-IID"
        new_details.append(cd)
    cfg["client_details"] = new_details
    return cfg


def copy_outputs(local_dir: Path, dst_dir: Path, model: str, dataset: str, method: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config used for the run.
    cfg_src = local_dir / "configuration" / "config.json"
    if cfg_src.exists():
        shutil.copy2(cfg_src, dst_dir / "config.used.json")

    # Save and rename only the final performance CSV.
    perf_dir = local_dir / "performance"
    src_csv = perf_dir / "FLwithAP_performance_metrics.csv"
    if src_csv.exists():
        out_name = f"{sanitize_name(model)}_{sanitize_name(dataset)}_{sanitize_name(method)}.csv"
        shutil.copy2(src_csv, dst_dir / out_name)
    else:
        # Fallback: if naming changes, pick the first matching CSV.
        csvs = sorted(perf_dir.glob("FLwithAP_performance_metrics*.csv"))
        if csvs:
            out_name = f"{sanitize_name(model)}_{sanitize_name(dataset)}_{sanitize_name(method)}.csv"
            shutil.copy2(csvs[0], dst_dir / out_name)


def main() -> int:
    root = Path(__file__).resolve().parent
    local_dir = root / "Local"
    config_path = local_dir / "configuration" / "config.json"

    ap = argparse.ArgumentParser(
        description="Run a Local AP4Fed experiment matrix (dataset x model x SSIM method)."
    )
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--criteria", default=",".join(DEFAULT_CRITERIA))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--num-supernodes", type=int, default=None)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    original_cfg = load_json(config_path)

    methods = parse_csv_arg(args.methods) if args.methods else list(DEFAULT_METHODS)
    models = parse_csv_arg(args.models) if args.models else list(DEFAULT_MODELS)
    datasets = parse_csv_arg(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    criteria_list = parse_csv_arg(args.criteria) if args.criteria else list(DEFAULT_CRITERIA)

    if not methods:
        print("No methods to run.", file=sys.stderr)
        return 2
    if not models:
        print("No models to run.", file=sys.stderr)
        return 2
    if not datasets:
        print("No datasets to run.", file=sys.stderr)
        return 2
    if not criteria_list:
        print("No selection criteria to run.", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    matrix: List[Tuple[str, str, str, str, int]] = []
    for dataset, model, method, criteria in itertools.product(datasets, models, methods, criteria_list):
        for r in range(1, args.repeat + 1):
            matrix.append((dataset, model, method, criteria, r))

    total = len(matrix)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = root / "Experiments" / "local_batch" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Total experiments: {total}")
    print(f"Output dir: {out_root}")

    failures = 0
    try:
        for idx, (dataset, model, method, criteria, repeat_idx) in enumerate(matrix, start=1):
            label = experiment_label(dataset, model, method, criteria, repeat_idx)
            print(
                f"\n[{idx}/{total}] Running "
                f"dataset='{dataset}', model='{model}', method='{method}', criteria='{criteria}', repeat={repeat_idx}"
            )

            cfg = apply_experiment_to_config(
                original_cfg, dataset, model, method, criteria, args.rounds
            )
            save_json(config_path, cfg)
            run_supernodes = int(args.num_supernodes) if args.num_supernodes is not None else int(cfg.get("clients", 1))

            if args.dry_run:
                print(f"[{idx}/{total}] DRY-RUN config prepared: {label} (supernodes={run_supernodes})")
                continue

            cmd = [
                "flower-simulation",
                "--app",
                ".",
                "--num-supernodes",
                str(run_supernodes),
            ]
            gpu_count = detect_cuda_gpu_count()
            if gpu_count > 0:
                if run_supernodes > gpu_count:
                    run_supernodes = gpu_count
                    cmd[4] = str(run_supernodes)
                    print(
                        f"[{idx}/{total}] CUDA detected: {gpu_count} GPU(s). "
                        f"Reducing supernodes to {run_supernodes} to avoid GPU contention."
                    )
                else:
                    print(f"[{idx}/{total}] CUDA detected: {gpu_count} GPU(s). Enabling GPU backend.")

                backend_cfg = {
                    "init_args": {"num_gpus": float(gpu_count)},
                    "client_resources": {"num_cpus": 1.0, "num_gpus": 1.0},
                }
                cmd.extend(["--backend-config", json.dumps(backend_cfg, separators=(",", ":"))])
            else:
                print(f"[{idx}/{total}] No CUDA GPU detected. Running Local simulation on CPU.")

            started_at = time.time()
            proc = subprocess.run(cmd, cwd=str(local_dir), check=False)
            elapsed = time.time() - started_at

            if proc.returncode != 0:
                failures += 1
                print(
                    f"[{idx}/{total}] FAILED (exit={proc.returncode}, elapsed={elapsed:.1f}s): {label}",
                    file=sys.stderr,
                )
                copy_outputs(local_dir, out_root / label, model, dataset, method)
                if not args.continue_on_error:
                    return 1
            else:
                print(f"[{idx}/{total}] OK (elapsed={elapsed:.1f}s): {label}")
                copy_outputs(local_dir, out_root / label, model, dataset, method)
    finally:
        # Always restore original config.
        save_json(config_path, original_cfg)
        print(f"\nRestored original config: {config_path}")

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
