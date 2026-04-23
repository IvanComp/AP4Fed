#!/usr/bin/env python3
"""Run the 6 RQ2 baseline/configuration tests locally without Docker.

This runner uses the Local Flower simulation but temporarily reuses the richer
agentic adaptation logic from Docker/adaptation.py so that Random, Voting,
Role-Based, Debate-Based, and Single-Agent policies behave consistently during
local experiments as well.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
LOCAL_DIR = ROOT / "Local"
LOCAL_CONFIG_PATH = LOCAL_DIR / "configuration" / "config.json"
LOCAL_ADAPTATION_PATH = LOCAL_DIR / "adaptation.py"
DOCKER_ADAPTATION_PATH = ROOT / "Docker" / "adaptation.py"
OUTPUT_DIR_DEFAULT = ROOT / "paper_results_local"

INDEX_FIELDS = [
    "Configuration",
    "Adaptation",
    "LLM",
    "Repeat",
    "Label",
    "Status",
    "Duration Seconds",
    "Output Dir",
    "ML Summary CSV",
    "Rationale CSV",
]


RQ2_CONFIGS = [
    {
        "name": "never",
        "adaptation": "None",
        "llm": "deepseek-r1:8b",
        "description": "Baseline Never",
    },
    {
        "name": "random",
        "adaptation": "Random",
        "llm": "deepseek-r1:8b",
        "description": "Baseline Random",
    },
    {
        "name": "expert_driven",
        "adaptation": "Expert-Driven",
        "llm": "deepseek-r1:8b",
        "description": "Baseline Expert-Driven",
    },
    {
        "name": "voting_based",
        "adaptation": "Multiple AI-Agents (Voting-Based)",
        "llm": "deepseek-r1:8b",
        "description": "Voting-based coordination",
    },
    {
        "name": "role_based",
        "adaptation": "Multiple AI-Agents (Role-Based)",
        "llm": "deepseek-r1:8b",
        "description": "Role-based coordination",
    },
    {
        "name": "debate_based",
        "adaptation": "Multiple AI-Agents (Debate-Based)",
        "llm": "deepseek-r1:8b",
        "description": "Debate-based coordination",
    },
]


CLIENT_TEMPLATE = [
    {
        "client_id": 1,
        "cpu": 5,
        "ram": 2,
        "dataset": "FashionMNIST",
        "data_distribution_type": "non-IID",
        "non_iid_alpha": 0.9,
        "data_persistence_type": "Same Data",
        "delay_combobox": "No",
        "delay_min_seconds": 0,
        "delay_max_seconds": 0,
        "model": "CNN 16k",
        "epochs": 1,
    },
    {
        "client_id": 2,
        "cpu": 5,
        "ram": 2,
        "dataset": "FashionMNIST",
        "data_distribution_type": "non-IID",
        "non_iid_alpha": 0.9,
        "data_persistence_type": "New Data",
        "delay_combobox": "No",
        "delay_min_seconds": 0,
        "delay_max_seconds": 0,
        "model": "CNN 16k",
        "epochs": 1,
    },
    {
        "client_id": 3,
        "cpu": 5,
        "ram": 2,
        "dataset": "FashionMNIST",
        "data_distribution_type": "IID",
        "non_iid_alpha": 0.5,
        "data_persistence_type": "New Data",
        "delay_combobox": "Yes",
        "delay_min_seconds": 20,
        "delay_max_seconds": 50,
        "model": "CNN 16k",
        "epochs": 1,
    },
    {
        "client_id": 4,
        "cpu": 3,
        "ram": 2,
        "dataset": "FashionMNIST",
        "data_distribution_type": "non-IID",
        "non_iid_alpha": 0.9,
        "data_persistence_type": "New Data",
        "delay_combobox": "Yes",
        "delay_min_seconds": 20,
        "delay_max_seconds": 50,
        "model": "CNN 16k",
        "epochs": 1,
    },
    {
        "client_id": 5,
        "cpu": 3,
        "ram": 2,
        "dataset": "FashionMNIST",
        "data_distribution_type": "IID",
        "non_iid_alpha": 0.5,
        "data_persistence_type": "Same Data",
        "delay_combobox": "No",
        "delay_min_seconds": 0,
        "delay_max_seconds": 0,
        "model": "CNN 16k",
        "epochs": 1,
    },
]


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def build_label(spec: Dict[str, str], repeat_idx: int) -> str:
    return f"local__rq2__{sanitize_name(spec['name'])}__r{repeat_idx:02d}"


def build_partition_seed(spec: Dict[str, str], repeat_idx: int, rounds: int) -> int:
    payload = json.dumps(
        {
            "mode": "local",
            "name": spec["name"],
            "adaptation": spec["adaptation"],
            "llm": spec["llm"],
            "repeat": repeat_idx,
            "rounds": rounds,
        },
        sort_keys=True,
    ).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def parse_csv_arg(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_patterns() -> Dict[str, Dict[str, Any]]:
    return {
        "client_registry": {"enabled": True, "params": {}},
        "client_selector": {
            "enabled": False,
            "params": {
                "selection_strategy": "Resource-Based",
                "selection_criteria": "CPU",
                "selection_value": 2,
            },
        },
        "client_cluster": {"enabled": False, "params": {}},
        "message_compressor": {"enabled": False, "params": {}},
        "model_co-versioning_registry": {"enabled": False, "params": {}},
        "multi-task_model_trainer": {"enabled": False, "params": {}},
        "heterogeneous_data_handler": {"enabled": False, "params": {}},
    }


def build_config(spec: Dict[str, str], rounds: int, repeat_idx: int, ollama_base_url: str) -> Dict[str, Any]:
    return {
        "simulation_type": "Local",
        "rounds": int(rounds),
        "clients": len(CLIENT_TEMPLATE),
        "clients_per_round": len(CLIENT_TEMPLATE),
        "dataset": "FashionMNIST",
        "adaptation": spec["adaptation"],
        "LLM": spec["llm"],
        "ollama_base_url": ollama_base_url,
        "partition_seed": build_partition_seed(spec, repeat_idx, rounds),
        "patterns": build_patterns(),
        "client_generation_mode": "manual",
        "client_profiles": [],
        "client_details": copy.deepcopy(CLIENT_TEMPLATE),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def restore_text_file(path: Path, original_text: Optional[str]) -> None:
    if original_text is None:
        path.unlink(missing_ok=True)
        return
    path.write_text(original_text, encoding="utf-8")


def reset_local_outputs() -> None:
    for folder_name in ("performance", "performance_MLdata", "logs"):
        folder_path = LOCAL_DIR / folder_name
        if not folder_path.exists():
            continue
        for entry in folder_path.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


def run_command(cmd: List[str], cwd: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(proc.returncode)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def archive_run_outputs(run_dir: Path, initial_config: Dict[str, Any]) -> Dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.initial.json", initial_config)
    copy_if_exists(LOCAL_CONFIG_PATH, run_dir / "config.final.json")
    copy_if_exists(LOCAL_DIR / "performance", run_dir / "performance")
    copy_if_exists(LOCAL_DIR / "performance_MLdata", run_dir / "performance_MLdata")
    copy_if_exists(LOCAL_DIR / "logs", run_dir / "logs")

    ml_csv = run_dir / "performance_MLdata" / "FLwithAP_MLdata.csv"
    rationale_csv = run_dir / "performance" / "FLwithAP_adaptation_rationales.csv"
    return {
        "ml_summary_csv": str(ml_csv) if ml_csv.exists() else "",
        "rationale_csv": str(rationale_csv) if rationale_csv.exists() else "",
    }


def append_index_row(index_csv_path: Path, row: Dict[str, str]) -> None:
    index_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_csv_path.exists()
    with index_csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})


def select_specs(all_specs: List[Dict[str, str]], requested_names: Optional[List[str]]) -> List[Dict[str, str]]:
    if not requested_names:
        return all_specs
    requested = set(requested_names)
    filtered = [spec for spec in all_specs if spec["name"] in requested]
    missing = sorted(requested - {spec["name"] for spec in filtered})
    if missing:
        raise ValueError(f"Unknown configuration filter(s): {', '.join(missing)}")
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 6 local RQ2 baselines without Docker.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--configs", default="", help="Optional comma-separated configuration filters.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.rounds < 1:
        print("--rounds must be >= 1", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    try:
        specs = select_specs(RQ2_CONFIGS, parse_csv_arg(args.configs) if args.configs else None)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_csv_path = output_dir / "index.csv"

    matrix: List[Dict[str, Any]] = []
    for spec in specs:
        for repeat_idx in range(1, args.repeat + 1):
            label = build_label(spec, repeat_idx)
            run_dir = output_dir / label
            matrix.append({"spec": spec, "repeat": repeat_idx, "label": label, "run_dir": run_dir})

    print(f"Planned local runs: {len(matrix)}")
    for item in matrix:
        spec = item["spec"]
        print(
            f"- {item['label']}: adaptation='{spec['adaptation']}' "
            f"llm='{spec['llm']}' -> {item['run_dir']}"
        )
    if args.dry_run:
        return 0

    original_config_text = read_text_if_exists(LOCAL_CONFIG_PATH)
    original_adaptation_text = read_text_if_exists(LOCAL_ADAPTATION_PATH)

    if not DOCKER_ADAPTATION_PATH.exists():
        print(f"Missing adaptation source: {DOCKER_ADAPTATION_PATH}", file=sys.stderr)
        return 2

    failures = 0
    try:
        shutil.copy2(DOCKER_ADAPTATION_PATH, LOCAL_ADAPTATION_PATH)
        for idx, item in enumerate(matrix, start=1):
            spec = item["spec"]
            repeat_idx = item["repeat"]
            label = item["label"]
            run_dir: Path = item["run_dir"]

            print(f"[{idx}/{len(matrix)}] Running {label} ({spec['description']})")
            reset_local_outputs()

            config = build_config(spec, rounds=args.rounds, repeat_idx=repeat_idx, ollama_base_url=args.ollama_base_url)
            write_json(LOCAL_CONFIG_PATH, config)

            env = dict(os.environ)
            env["AP4FED_ROUNDS_OVERRIDE"] = str(args.rounds)
            env["AGENT_LOG_TO_FILE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            log_path = run_dir / "flower.log"
            started_at = time.time()
            rc = run_command(
                ["flower-simulation", "--app", ".", "--num-supernodes", str(config["clients"])],
                cwd=LOCAL_DIR,
                log_path=log_path,
                env=env,
            )
            duration_seconds = f"{time.time() - started_at:.1f}"

            archived = archive_run_outputs(run_dir, config)
            status = "ok" if rc == 0 else f"failed({rc})"
            append_index_row(
                index_csv_path,
                {
                    "Configuration": spec["name"],
                    "Adaptation": spec["adaptation"],
                    "LLM": spec["llm"],
                    "Repeat": str(repeat_idx),
                    "Label": label,
                    "Status": status,
                    "Duration Seconds": duration_seconds,
                    "Output Dir": str(run_dir),
                    "ML Summary CSV": archived["ml_summary_csv"],
                    "Rationale CSV": archived["rationale_csv"],
                },
            )

            if rc != 0:
                failures += 1
                print(f"[{idx}/{len(matrix)}] FAILED: {label} (exit={rc})", file=sys.stderr)
                if not args.continue_on_error:
                    return rc or 1
            else:
                print(f"[{idx}/{len(matrix)}] OK: {label} ({duration_seconds}s)")
    finally:
        restore_text_file(LOCAL_CONFIG_PATH, original_config_text)
        restore_text_file(LOCAL_ADAPTATION_PATH, original_adaptation_text)

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("All local runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
