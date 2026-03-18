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


DEFAULT_MODELS = ["CNN 16k"]
DEFAULT_DATASETS = ["CIFAR-10", "FashionMNIST"]
DEFAULT_EXPERIMENTS = ["baseline", "client_selector", "heterogeneous_data_handler", "message_compressor"]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def parse_csv_arg(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def detect_cuda_gpu_count() -> int:
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


def sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")


def experiment_label(dataset: str, model: str, experiment_name: str, repeat_idx: int) -> str:
    safe = sanitize_name
    return f"{safe(dataset)}__{safe(model)}__{safe(experiment_name)}__r{repeat_idx:02d}"


def reset_patterns(cfg: Dict[str, Any]) -> None:
    for pattern_cfg in cfg.setdefault("patterns", {}).values():
        pattern_cfg["enabled"] = False
        pattern_cfg["params"] = pattern_cfg.get("params", {}) or {}

    cfg["patterns"].setdefault("client_registry", {"enabled": True, "params": {}})
    cfg["patterns"]["client_registry"]["enabled"] = True


def build_client_details(template: Dict[str, Any], dataset: str, model: str) -> List[Dict[str, Any]]:
    client_details = []
    for cid in range(1, 6):
        cd = copy.deepcopy(template)
        cd["client_id"] = cid
        cd["cpu"] = 1 if cid >= 4 else 2
        cd["ram"] = 2
        cd["dataset"] = dataset
        cd["model"] = model
        cd["epochs"] = int(cd.get("epochs", 1))
        cd["delay_combobox"] = cd.get("delay_combobox", "No")
        cd["data_persistence_type"] = cd.get("data_persistence_type", "Same Data")
        cd["data_distribution_type"] = "IID" if cid <= 3 else "non-IID"
        client_details.append(cd)
    return client_details


def apply_experiment_to_config(
    base_cfg: Dict[str, Any],
    dataset: str,
    model: str,
    experiment_name: str,
    rounds_override: Optional[int],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    if rounds_override is not None:
        cfg["rounds"] = int(rounds_override)

    cfg["simulation_type"] = "Local"
    cfg["adaptation"] = "None"
    cfg["clients"] = 5
    cfg["clients_per_round"] = 5
    cfg["client_generation_mode"] = "manual"
    cfg["client_profiles"] = []

    reset_patterns(cfg)

    existing = cfg.get("client_details", [])
    template = existing[0] if existing else {}
    cfg["client_details"] = build_client_details(template, dataset, model)

    if experiment_name == "baseline":
        pass
    elif experiment_name == "client_selector":
        selector = cfg["patterns"].setdefault("client_selector", {"enabled": False, "params": {}})
        selector["enabled"] = True
        selector["params"] = {
            "selection_strategy": "Resource-Based",
            "selection_criteria": "CPU",
            "selection_value": 2,
        }
    elif experiment_name == "heterogeneous_data_handler":
        hdh = cfg["patterns"].setdefault("heterogeneous_data_handler", {"enabled": False, "params": {}})
        hdh["enabled"] = True
        hdh["params"] = {}
    elif experiment_name == "message_compressor":
        mc = cfg["patterns"].setdefault("message_compressor", {"enabled": False, "params": {}})
        mc["enabled"] = True
        mc["params"] = {}
    else:
        raise ValueError(f"Unknown experiment configuration: {experiment_name}")

    return cfg


def copy_outputs(local_dir: Path, dst_dir: Path, model: str, dataset: str, experiment_name: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    cfg_src = local_dir / "configuration" / "config.json"
    if cfg_src.exists():
        shutil.copy2(cfg_src, dst_dir / "config.used.json")

    perf_dir = local_dir / "performance"
    src_csv = perf_dir / "FLwithAP_performance_metrics.csv"
    if src_csv.exists():
        out_name = f"{sanitize_name(model)}_{sanitize_name(dataset)}_{sanitize_name(experiment_name)}.csv"
        shutil.copy2(src_csv, dst_dir / out_name)

    ml_dir = local_dir / "performance_MLdata"
    src_ml_csv = ml_dir / "FLwithAP_MLdata.csv"
    if src_ml_csv.exists():
        out_name = f"{sanitize_name(model)}_{sanitize_name(dataset)}_{sanitize_name(experiment_name)}_MLdata.csv"
        shutil.copy2(src_ml_csv, dst_dir / out_name)


def main() -> int:
    root = Path(__file__).resolve().parent
    local_dir = root / "Local"
    config_path = local_dir / "configuration" / "config.json"

    ap = argparse.ArgumentParser(
        description="Run a Local AP4Fed experiment matrix (dataset x model x pattern configuration)."
    )
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--num-supernodes", type=int, default=None)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    original_cfg = load_json(config_path)

    models = parse_csv_arg(args.models) if args.models else list(DEFAULT_MODELS)
    datasets = parse_csv_arg(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    experiments = parse_csv_arg(args.experiments) if args.experiments else list(DEFAULT_EXPERIMENTS)

    if not models:
        print("No models to run.", file=sys.stderr)
        return 2
    if not datasets:
        print("No datasets to run.", file=sys.stderr)
        return 2
    if not experiments:
        print("No experiment configurations to run.", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    matrix: List[Tuple[str, str, str, int]] = []
    for dataset, model, experiment_name in itertools.product(datasets, models, experiments):
        for r in range(1, args.repeat + 1):
            matrix.append((dataset, model, experiment_name, r))

    total = len(matrix)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = root / "Experiments" / "local_batch" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Total experiments: {total}")
    print(f"Output dir: {out_root}")

    failures = 0
    try:
        for idx, (dataset, model, experiment_name, repeat_idx) in enumerate(matrix, start=1):
            label = experiment_label(dataset, model, experiment_name, repeat_idx)
            print(
                f"\n[{idx}/{total}] Running "
                f"dataset='{dataset}', model='{model}', config='{experiment_name}', repeat={repeat_idx}"
            )

            cfg = apply_experiment_to_config(
                original_cfg, dataset, model, experiment_name, args.rounds
            )
            save_json(config_path, cfg)

            configured_clients = int(cfg.get("clients", 1))
            requested_supernodes = (
                int(args.num_supernodes) if args.num_supernodes is not None else configured_clients
            )
            run_supernodes = max(requested_supernodes, configured_clients)
            if run_supernodes != requested_supernodes:
                print(
                    f"[{idx}/{total}] Increasing supernodes from {requested_supernodes} to {run_supernodes} "
                    f"to match configured clients ({configured_clients})."
                )

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
                per_client_gpus = min(1.0, float(gpu_count) / float(run_supernodes))
                per_client_gpus = max(per_client_gpus, 0.01)
                print(
                    f"[{idx}/{total}] CUDA detected: {gpu_count} GPU(s). "
                    f"Using {run_supernodes} supernode(s) with {per_client_gpus:.2f} GPU per client."
                )
                backend_cfg = {
                    "init_args": {"num_gpus": float(gpu_count)},
                    "client_resources": {"num_cpus": 1.0, "num_gpus": per_client_gpus},
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
                copy_outputs(local_dir, out_root / label, model, dataset, experiment_name)
                if not args.continue_on_error:
                    return 1
            else:
                print(f"[{idx}/{total}] OK (elapsed={elapsed:.1f}s): {label}")
                copy_outputs(local_dir, out_root / label, model, dataset, experiment_name)
    finally:
        save_json(config_path, original_cfg)

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1

    print("\nAll experiments completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
