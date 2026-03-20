#!/usr/bin/env python3
import argparse
import csv
import copy
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODELS = ["CNN 16k"]
DEFAULT_DATASETS = ["FashionMNIST", "CIFAR-10"]
DEFAULT_EXPERIMENTS = ["baseline", "client_selector", "heterogeneous_data_handler", "message_compressor"]
DEFAULT_CLIENT_SETUPS = [
    "manual5",
    "profile_34_33_33",
    "profile_60_30_10",
    "profile_60_10_30",
    "profile_30_60_10",
    "profile_10_60_30",
    "profile_30_10_60",
    "profile_10_30_60",
]
PROFILE_SHARE_MAP = {
    "profile_34_33_33": (34, 33, 33),
    "profile_60_30_10": (60, 30, 10),
    "profile_60_10_30": (60, 10, 30),
    "profile_30_60_10": (30, 60, 10),
    "profile_10_60_30": (10, 60, 30),
    "profile_30_10_60": (30, 10, 60),
    "profile_10_30_60": (10, 30, 60),
}


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
        import torch 

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass

    return 0


def sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s).strip("_")


def experiment_label(
    dataset: str,
    model: str,
    experiment_name: str,
    repeat_idx: int,
    client_setup: str = "manual5",
) -> str:
    safe = sanitize_name
    base = f"{safe(dataset)}__{safe(model)}__{safe(experiment_name)}"
    if client_setup != "manual5":
        base += f"__{safe(client_setup)}"
    return f"{base}__r{repeat_idx:02d}"


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
        if cid in (1, 2):
            cd["cpu"] = 3
        elif cid == 3:
            cd["cpu"] = 2
        else:
            cd["cpu"] = 1
        cd["ram"] = 2
        cd["dataset"] = dataset
        cd["model"] = model
        cd["epochs"] = int(cd.get("epochs", 1))
        cd["delay_combobox"] = cd.get("delay_combobox", "No")
        cd["data_persistence_type"] = cd.get("data_persistence_type", "Same Data")
        cd["data_distribution_type"] = "IID" if cid <= 3 else "non-IID"
        client_details.append(cd)
    return client_details


def normalize_profile_counts(total_clients: int, shares: List[int]) -> List[int]:
    if total_clients <= 0 or not shares or sum(shares) <= 0:
        return [0 for _ in shares]
    raw = [(share / sum(shares)) * total_clients for share in shares]
    floors = [int(value) for value in raw]
    counts = floors[:]
    leftover = total_clients - sum(floors)
    fractions = [value - floor for value, floor in zip(raw, floors)]
    order = sorted(range(len(shares)), key=lambda idx: fractions[idx], reverse=True)
    for idx in order[:leftover]:
        counts[idx] += 1
    return counts


def build_profile_scenario(
    template: Dict[str, Any],
    dataset: str,
    model: str,
    client_setup: str,
    experiment_name: str,
    repeat_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    shares = PROFILE_SHARE_MAP[client_setup]
    total_clients = 100
    clients_per_round = 5
    counts = normalize_profile_counts(total_clients, list(shares))
    profiles = [
        {
            "profile_id": 1,
            "share_percent": shares[0],
            "cpu": 3,
            "ram": 2,
            "dataset": dataset,
            "data_distribution_type": "Random",
            "data_persistence_type": "Same Data",
            "delay_combobox": "No",
            "model": model,
            "epochs": int(template.get("epochs", 1) or 1),
        },
        {
            "profile_id": 2,
            "share_percent": shares[1],
            "cpu": 2,
            "ram": 2,
            "dataset": dataset,
            "data_distribution_type": "Random",
            "data_persistence_type": "Same Data",
            "delay_combobox": "No",
            "model": model,
            "epochs": int(template.get("epochs", 1) or 1),
        },
        {
            "profile_id": 3,
            "share_percent": shares[2],
            "cpu": 1,
            "ram": 2,
            "dataset": dataset,
            "data_distribution_type": "Random",
            "data_persistence_type": "Same Data",
            "delay_combobox": "No",
            "model": model,
            "epochs": int(template.get("epochs", 1) or 1),
        },
    ]

    rng = json.dumps(
        {
            "dataset": dataset,
            "model": model,
            "experiment": experiment_name,
            "repeat": repeat_idx,
            "setup": client_setup,
        },
        sort_keys=True,
    )
    seeded = __import__("random").Random(rng)

    client_details: List[Dict[str, Any]] = []
    next_client_id = 1
    for profile, count in zip(profiles, counts):
        for _ in range(count):
            cd = copy.deepcopy(template)
            cd["client_id"] = next_client_id
            cd["cpu"] = profile["cpu"]
            cd["ram"] = profile["ram"]
            cd["dataset"] = dataset
            cd["data_distribution_type"] = seeded.choice(["IID", "non-IID"])
            cd["data_persistence_type"] = "Same Data"
            cd["delay_combobox"] = "No"
            cd["model"] = model
            cd["epochs"] = int(template.get("epochs", 1) or 1)
            client_details.append(cd)
            next_client_id += 1

    return client_details, profiles, total_clients, clients_per_round


def apply_experiment_to_config(
    base_cfg: Dict[str, Any],
    dataset: str,
    model: str,
    experiment_name: str,
    rounds_override: Optional[int],
    client_setup: str,
    repeat_idx: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    if rounds_override is not None:
        cfg["rounds"] = int(rounds_override)

    cfg["simulation_type"] = "Local"
    cfg["adaptation"] = "None"

    reset_patterns(cfg)

    existing = cfg.get("client_details", [])
    template = existing[0] if existing else {}
    if client_setup == "manual5":
        cfg["clients"] = 5
        cfg["clients_per_round"] = 5
        cfg["client_generation_mode"] = "manual"
        cfg["client_profiles"] = []
        cfg["client_details"] = build_client_details(template, dataset, model)
    else:
        client_details, profiles, total_clients, clients_per_round = build_profile_scenario(
            template,
            dataset,
            model,
            client_setup,
            experiment_name,
            repeat_idx,
        )
        cfg["clients"] = total_clients
        cfg["clients_per_round"] = clients_per_round
        cfg["client_generation_mode"] = "profile_based"
        cfg["client_profiles"] = profiles
        cfg["client_details"] = client_details

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


def reset_experiment_outputs(local_dir: Path) -> None:
    for rel in [
        ("performance", "FLwithAP_performance_metrics.csv"),
        ("performance_MLdata", "FLwithAP_MLdata.csv"),
    ]:
        path = local_dir / rel[0] / rel[1]
        if path.exists():
            path.unlink()


def infer_experiment_name_from_ap_list(ap_list_value: str) -> Optional[str]:
    normalized = str(ap_list_value or "").strip().upper().replace(" ", "")
    mapping = {
        "{OFF,OFF,OFF}": "baseline",
        "{ON,OFF,OFF}": "client_selector",
        "{OFF,ON,OFF}": "message_compressor",
        "{OFF,OFF,ON}": "heterogeneous_data_handler",
    }
    return mapping.get(normalized)


def get_batch_fieldnames(source_fieldnames: List[str]) -> List[str]:
    metadata_fields = [
        "Runall Dataset",
        "Runall Model",
        "Runall Config",
        "Runall Client Setup",
        "Runall Repeat",
        "Runall Rounds",
        "Runall Label",
    ]
    return metadata_fields + source_fieldnames


def convert_row_to_current_batch_schema(
    row: Dict[str, str],
    expected_fieldnames: List[str],
) -> Dict[str, str]:
    converted = {field: "" for field in expected_fieldnames}

    for field in expected_fieldnames:
        if field in row:
            converted[field] = row.get(field, "")

    if not converted["Runall Dataset"]:
        converted["Runall Dataset"] = row.get("Dataset", "")
    if not converted["Runall Model"]:
        converted["Runall Model"] = row.get("Model", "")
    if not converted["Runall Rounds"]:
        converted["Runall Rounds"] = row.get("N Rounds", "")

    if not converted["Runall Config"]:
        converted["Runall Config"] = infer_experiment_name_from_ap_list(
            row.get("AP List (client_selector,message_compressor,heterogeneous_data_handler)", "")
        ) or ""

    if not converted["Runall Client Setup"]:
        converted["Runall Client Setup"] = "manual5"

    if not converted["Runall Repeat"]:
        converted["Runall Repeat"] = "1"

    if not converted["Runall Label"]:
        dataset = converted["Runall Dataset"]
        model = converted["Runall Model"]
        experiment_name = converted["Runall Config"]
        client_setup = converted["Runall Client Setup"] or "manual5"
        repeat_raw = converted["Runall Repeat"] or "1"
        try:
            repeat_idx = int(float(str(repeat_raw).replace(",", ".")))
        except Exception:
            repeat_idx = 1
        if dataset and model and experiment_name:
            converted["Runall Label"] = experiment_label(
                dataset,
                model,
                experiment_name,
                repeat_idx,
                client_setup,
            )

    return converted


def ensure_batch_csv_schema(batch_csv_path: Path, expected_fieldnames: List[str]) -> None:
    if not batch_csv_path.exists():
        return

    with batch_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        existing_fieldnames = reader.fieldnames or []
        rows = list(reader)

    if existing_fieldnames == expected_fieldnames:
        return

    converted_rows = [
        convert_row_to_current_batch_schema(row, expected_fieldnames)
        for row in rows
    ]

    with batch_csv_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=expected_fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(converted_rows)


def read_completed_labels(batch_csv_path: Path, expected_rounds: int) -> set[str]:
    if not batch_csv_path.exists():
        return set()

    raw_lines = [line for line in batch_csv_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if len(raw_lines) <= 1:
        return set()

    completed = set()

    for line in raw_lines[1:]:
        parts = line.split(";")
        if len(parts) < 6:
            continue

        # Newer runall rows:
        # dataset, model, config, client_setup, repeat, rounds, label, ...
        # Older runall rows:
        # dataset, model, config, repeat, rounds, label, ...
        if len(parts) >= 7 and not str(parts[3]).isdigit():
            client_setup = parts[3].strip() or "manual5"
            row_rounds = parts[5].strip()
            label = parts[6].strip()
        else:
            client_setup = "manual5"
            row_rounds = parts[4].strip()
            label = parts[5].strip()

        try:
            if label and int(float(row_rounds.replace(",", "."))) == expected_rounds:
                completed.add(label)
        except Exception:
            continue

    return completed


def append_to_batch_ml_summary(
    local_dir: Path,
    batch_csv_path: Path,
    dataset: str,
    model: str,
    experiment_name: str,
    repeat_idx: int,
    label: str,
    client_setup: str,
) -> bool:
    src_ml_csv = local_dir / "performance_MLdata" / "FLwithAP_MLdata.csv"
    if not src_ml_csv.exists():
        return False

    with src_ml_csv.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f, delimiter=";"))
        if not reader:
            return False
        source_fieldnames = list(reader[0].keys())
        last_row = reader[-1]

    metadata = {
        "Runall Dataset": dataset,
        "Runall Model": model,
        "Runall Config": experiment_name,
        "Runall Client Setup": client_setup,
        "Runall Repeat": repeat_idx,
        "Runall Rounds": last_row.get("N Rounds"),
        "Runall Label": label,
    }
    combined_row = {**metadata, **last_row}
    fieldnames = get_batch_fieldnames(source_fieldnames)

    ensure_batch_csv_schema(batch_csv_path, fieldnames)

    write_header = not batch_csv_path.exists()
    with batch_csv_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(combined_row)
    return True


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
    ap.add_argument("--client-setups", default=",".join(DEFAULT_CLIENT_SETUPS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=100)
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
    client_setups = parse_csv_arg(args.client_setups) if args.client_setups else list(DEFAULT_CLIENT_SETUPS)

    if not models:
        print("No models to run.", file=sys.stderr)
        return 2
    if not datasets:
        print("No datasets to run.", file=sys.stderr)
        return 2
    if not experiments:
        print("No experiment configurations to run.", file=sys.stderr)
        return 2
    if not client_setups:
        print("No client setups to run.", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    ordered_client_setups = []
    if "manual5" in client_setups:
        ordered_client_setups.append("manual5")
    ordered_client_setups.extend([setup for setup in client_setups if setup != "manual5"])

    matrix: List[Tuple[str, str, str, str, int]] = []

    if "manual5" in ordered_client_setups:
        for dataset, model, experiment_name in itertools.product(datasets, models, experiments):
            for r in range(1, args.repeat + 1):
                matrix.append((dataset, model, experiment_name, "manual5", r))

    profile_setups = [setup for setup in ordered_client_setups if setup != "manual5"]
    for dataset in datasets:
        for client_setup in profile_setups:
            for model, experiment_name in itertools.product(models, experiments):
                for r in range(1, args.repeat + 1):
                    matrix.append((dataset, model, experiment_name, client_setup, r))

    total = len(matrix)
    batch_ml_csv = local_dir / "performance_MLdata" / "FLwithAP_MLdata_runall.csv"
    batch_ml_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_labels = read_completed_labels(batch_ml_csv, args.rounds)

    print(f"Total experiments: {total}")

    failures = 0
    try:
        for idx, (dataset, model, experiment_name, client_setup, repeat_idx) in enumerate(matrix, start=1):
            label = experiment_label(dataset, model, experiment_name, repeat_idx, client_setup)
            if label in completed_labels:
                print(f"[{idx}/{total}] Skipping completed: {label}")
                continue
            print(
                f"\n[{idx}/{total}] Running "
                f"dataset='{dataset}', model='{model}', config='{experiment_name}', setup='{client_setup}', repeat={repeat_idx}"
            )

            cfg = apply_experiment_to_config(
                original_cfg, dataset, model, experiment_name, args.rounds, client_setup, repeat_idx
            )
            save_json(config_path, cfg)
            reset_experiment_outputs(local_dir)

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
                print(f"[{idx}/{total}] DRY-RUN config prepared: {label} (supernodes={run_supernodes}, rounds={cfg['rounds']}, clients={cfg['clients']}, k={cfg['clients_per_round']})")
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

            run_env = dict(os.environ)
            run_env["AP4FED_ROUNDS_OVERRIDE"] = str(cfg["rounds"])

            log_dir = local_dir / "performance" / "runall_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{label}.log"

            started_at = time.time()
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    cmd,
                    cwd=str(local_dir),
                    check=False,
                    env=run_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            elapsed = time.time() - started_at

            if proc.returncode != 0:
                failures += 1
                print(
                    f"[{idx}/{total}] FAILED (exit={proc.returncode}, elapsed={elapsed:.1f}s): {label}",
                    file=sys.stderr,
                )
                continue
            else:
                appended = append_to_batch_ml_summary(
                    local_dir,
                    batch_ml_csv,
                    dataset,
                    model,
                    experiment_name,
                    repeat_idx,
                    label,
                    client_setup,
                )
                if not appended:
                    failures += 1
                    print(
                        f"[{idx}/{total}] FAILED (missing ML CSV, elapsed={elapsed:.1f}s): {label}",
                        file=sys.stderr,
                    )
                    continue

                completed_labels.add(label)
                print(f"[{idx}/{total}] OK (elapsed={elapsed:.1f}s): {label}")
    finally:
        save_json(config_path, original_cfg)

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1

    print("\nAll experiments completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
