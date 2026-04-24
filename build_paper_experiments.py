#!/usr/bin/env python3
"""Single entry point for the paper's 6-approach local campaign.

The script can:
1. Run the 6 updated local experiments.
2. Export the outputs into the historical Experiments layout.
3. Regenerate the main paper figures and the summary CSV.

By default it targets a 100-round, single-run campaign.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
LOCAL_DIR = ROOT / "Local"
DOCKER_DIR = ROOT / "Docker"
LOCAL_CONFIG_PATH = LOCAL_DIR / "configuration" / "config.json"
DOCKER_CONFIG_PATH = DOCKER_DIR / "configuration" / "config.json"
LOCAL_ADAPTATION_PATH = LOCAL_DIR / "adaptation.py"
DOCKER_ADAPTATION_PATH = ROOT / "Docker" / "adaptation.py"
LOCAL_OUTPUT_DIR = ROOT / "Experiments_100r"
LOCAL_STAGING_DIR = ROOT / "paper_results_local_100r"
DOCKER_OUTPUT_DIR = ROOT / "Experiments_100r_docker"
DOCKER_STAGING_DIR = ROOT / "paper_results_docker_100r"

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

plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["hatch.color"] = "#777777"
plt.rcParams["hatch.linewidth"] = 0.6
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class ApproachSpec:
    runner_name: str
    folder_name: str
    display_name: str
    adaptation: str
    llm: str
    description: str


APPROACH_SPECS: Tuple[ApproachSpec, ...] = (
    ApproachSpec("never", "never", "Never", "None", "deepseek-r1:8b", "Baseline Never"),
    ApproachSpec("random", "random", "Random", "Random", "deepseek-r1:8b", "Baseline Random"),
    ApproachSpec("expert_driven", "expert-driven", "Expert-Driven", "Expert-Driven", "deepseek-r1:8b", "Baseline Expert-Driven"),
    ApproachSpec("voting_based", "voting-based", "Voting-based", "Multiple AI-Agents (Voting-Based)", "deepseek-r1:8b", "Voting-based coordination"),
    ApproachSpec("role_based", "role-based", "Role-based", "Multiple AI-Agents (Role-Based)", "deepseek-r1:8b", "Role-based coordination"),
    ApproachSpec("debate_based", "debate-based", "Debate-based", "Multiple AI-Agents (Debate-Based)", "deepseek-r1:8b", "Debate-based coordination"),
)

DISPLAY_ORDER = [spec.display_name for spec in APPROACH_SPECS]
DISPLAY_TO_FOLDER = {spec.display_name: spec.folder_name for spec in APPROACH_SPECS}
RUNNER_TO_SPEC = {spec.runner_name: spec for spec in APPROACH_SPECS}
DISPLAY_COLORS = {
    "Never": "#b8b8b8",
    "Random": "#a0a0a0",
    "Expert-Driven": "#606060",
    "Voting-based": "#ffb482",
    "Role-based": "#8de5a1",
    "Debate-based": "#f6a6a6",
}
DISPLAY_LINESTYLES = {
    "Never": ":",
    "Random": "--",
    "Expert-Driven": "-.",
    "Voting-based": "-",
    "Role-based": "-",
    "Debate-based": "-",
}
DISPLAY_HATCHES = {
    "Never": "",
    "Random": "///",
    "Expert-Driven": "XXX",
    "Voting-based": "\\\\",
    "Role-based": "--",
    "Debate-based": "oo",
}

PATTERN_NAMES = ("CS", "MC", "HDH")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fresh Experiments-style folder for the paper's 6 approaches."
    )
    parser.add_argument("--mode", choices=("local", "docker"), default="local")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Target total number of runs per approach. Existing rN.csv files are reused and missing runs resume from the next index.",
    )
    parser.add_argument("--staging-dir")
    parser.add_argument("--ollama-base-url")
    parser.add_argument("--skip-run", action="store_true", help="Reuse an existing staging directory.")
    parser.add_argument("--force", action="store_true", help="Reset both staging and exported outputs, then start again from r1.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--pattern-run", type=int, default=1, help="Run id used for the pattern activation figure.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_empty_dir(path: Path, force: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not force:
                raise FileExistsError(
                    f"Directory '{path}' already exists and is not empty. "
                    f"Use --force or pick another destination."
                )
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def reset_dir_if_needed(path: Path, keep_existing: bool) -> None:
    if path.exists() and not keep_existing:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def build_label(spec: ApproachSpec, repeat_idx: int) -> str:
    return f"local__rq2__{sanitize_name(spec.runner_name)}__r{repeat_idx:02d}"


def build_partition_seed(spec: ApproachSpec, repeat_idx: int, rounds: int) -> int:
    payload = json.dumps(
        {
            "mode": "local",
            "name": spec.runner_name,
            "adaptation": spec.adaptation,
            "llm": spec.llm,
            "repeat": repeat_idx,
            "rounds": rounds,
        },
        sort_keys=True,
    ).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


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


def default_output_dir_for_mode(mode: str) -> Path:
    return LOCAL_OUTPUT_DIR if mode == "local" else DOCKER_OUTPUT_DIR


def default_staging_dir_for_mode(mode: str) -> Path:
    return LOCAL_STAGING_DIR if mode == "local" else DOCKER_STAGING_DIR


def default_ollama_url_for_mode(mode: str) -> str:
    return "http://127.0.0.1:11434" if mode == "local" else "http://host.docker.internal:11434"


def notebook_template_for_output(output_dir: Path) -> Path:
    base_name = output_dir.name.removesuffix("_docker")
    k5_variant = "k5" in base_name
    template_dir = ROOT / ("Experiments_100r_k5" if k5_variant else "Experiments_100r")
    return template_dir / "Paper Figures.ipynb"


def sync_output_notebook(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "Paper Figures.ipynb"
    template = notebook_template_for_output(output_dir)
    if not template.exists():
        return

    if template.resolve() == destination.resolve():
        return

    notebook = json.loads(template.read_text(encoding="utf-8"))
    root_line = f'ROOT = Path("{output_dir}")\n'
    for cell in notebook.get("cells", []):
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        for idx, line in enumerate(source):
            if isinstance(line, str) and line.strip().startswith("ROOT = Path("):
                source[idx] = root_line

    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def build_local_config(
    spec: ApproachSpec,
    rounds: int,
    repeat_idx: int,
    ollama_base_url: str,
    simulation_type: str = "Local",
) -> Dict[str, Any]:
    return {
        "simulation_type": simulation_type,
        "rounds": int(rounds),
        "clients": len(CLIENT_TEMPLATE),
        "clients_per_round": len(CLIENT_TEMPLATE),
        "dataset": "FashionMNIST",
        "adaptation": spec.adaptation,
        "LLM": spec.llm,
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


def reset_runtime_outputs(runtime_dir: Path) -> None:
    for folder_name in ("performance", "performance_MLdata", "logs", "model_weights"):
        folder_path = runtime_dir / folder_name
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


def archive_run_outputs(run_dir: Path, runtime_dir: Path, config_path: Path, initial_config: Dict[str, Any]) -> Dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.initial.json", initial_config)
    copy_if_exists(config_path, run_dir / "config.final.json")
    copy_if_exists(runtime_dir / "performance", run_dir / "performance")
    copy_if_exists(runtime_dir / "performance_MLdata", run_dir / "performance_MLdata")
    copy_if_exists(runtime_dir / "logs", run_dir / "logs")

    ml_csv = run_dir / "performance_MLdata" / "FLwithAP_MLdata.csv"
    rationale_csv = run_dir / "performance" / "FLwithAP_adaptation_rationales.csv"
    return {
        "ml_summary_csv": str(ml_csv) if ml_csv.exists() else "",
        "rationale_csv": str(rationale_csv) if rationale_csv.exists() else "",
    }


def append_stage_index_row(index_csv_path: Path, row: Dict[str, str]) -> None:
    index_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_csv_path.exists()
    with index_csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})


def existing_exported_repeats(output_dir: Path) -> Dict[str, set[int]]:
    repeats: Dict[str, set[int]] = {spec.runner_name: set() for spec in APPROACH_SPECS}
    for spec in APPROACH_SPECS:
        folder = output_dir / spec.folder_name
        if not folder.exists():
            continue
        for csv_path in folder.glob("r*.csv"):
            match = re.fullmatch(r"r(\d+)", csv_path.stem)
            if match:
                repeats[spec.runner_name].add(int(match.group(1)))
    return repeats


def existing_staging_repeats(staging_dir: Path) -> Dict[str, set[int]]:
    repeats: Dict[str, set[int]] = {spec.runner_name: set() for spec in APPROACH_SPECS}
    index_path = staging_dir / "index.csv"
    if not index_path.exists():
        return repeats

    df = pd.read_csv(index_path)
    if df.empty:
        return repeats

    if "Repeat" in df.columns:
        df["Repeat"] = pd.to_numeric(df["Repeat"], errors="coerce")

    for spec in APPROACH_SPECS:
        subset = df.loc[
            (df["Configuration"] == spec.runner_name)
            & (df["Status"].astype(str).str.startswith("ok"))
        ]
        valid = subset["Repeat"].dropna().astype(int).tolist() if "Repeat" in subset.columns else []
        repeats[spec.runner_name].update(valid)
    return repeats


def plan_resume_matrix(target_repeat: int, staging_dir: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[int]]]]:
    staged = existing_staging_repeats(staging_dir)
    exported = existing_exported_repeats(output_dir)

    matrix: List[Dict[str, Any]] = []
    plan: Dict[str, Dict[str, List[int]]] = {}
    for spec in APPROACH_SPECS:
        completed = sorted(staged.get(spec.runner_name, set()) | exported.get(spec.runner_name, set()))
        missing = [repeat_idx for repeat_idx in range(1, target_repeat + 1) if repeat_idx not in completed]
        plan[spec.runner_name] = {
            "completed": completed,
            "missing": missing,
        }
        for repeat_idx in missing:
            label = build_label(spec, repeat_idx)
            run_dir = staging_dir / label
            matrix.append({"spec": spec, "repeat": repeat_idx, "label": label, "run_dir": run_dir})

    return matrix, plan


def refresh_exported_outputs(staging_dir: Path, output_dir: Path, pattern_run_id: int) -> None:
    sync_output_notebook(output_dir)
    stage_df = read_stage_index(staging_dir)
    export_stage_to_experiments(stage_df, output_dir)
    try:
        generate_outputs(output_dir, pattern_run_id=pattern_run_id)
    except ValueError:
        # No successful runs have been exported yet.
        pass


def build_docker_compose(config: Dict[str, Any], compose_path: Path) -> None:
    template_path = DOCKER_DIR / "docker-compose.yml"
    with template_path.open("r", encoding="utf-8") as handle:
        compose = yaml.safe_load(handle)

    server_svc = copy.deepcopy(compose["services"].get("server"))
    client_tpl = copy.deepcopy(compose["services"].get("client"))
    if not server_svc or not client_tpl:
        raise ValueError("docker-compose.yml must define both 'server' and 'client' services")

    server_svc["image"] = "ap4fed_server:latest"
    server_env = server_svc.setdefault("environment", {})
    server_env["NUM_ROUNDS"] = str(config["rounds"])
    extra_hosts = server_svc.setdefault("extra_hosts", [])
    host_gateway_entry = "host.docker.internal:host-gateway"
    if host_gateway_entry not in extra_hosts:
        extra_hosts.append(host_gateway_entry)

    new_svcs = {"server": server_svc}
    for detail in config["client_details"]:
        cid = detail["client_id"]
        cpu = detail["cpu"]
        ram = detail["ram"]

        svc = copy.deepcopy(client_tpl)
        svc.pop("deploy", None)
        svc["image"] = "ap4fed_client:latest"
        svc["container_name"] = f"Client{cid}"
        svc["cpus"] = float(cpu)
        svc["mem_limit"] = f"{ram}g"

        env = svc.setdefault("environment", {})
        env["NUM_ROUNDS"] = str(config["rounds"])
        env["NUM_CPUS"] = str(cpu)
        env["NUM_RAM"] = f"{ram}g"
        env["CLIENT_ID"] = str(cid)

        extra_hosts = svc.setdefault("extra_hosts", [])
        if host_gateway_entry not in extra_hosts:
            extra_hosts.append(host_gateway_entry)

        new_svcs[f"client{cid}"] = svc

    compose["services"] = new_svcs
    with compose_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(compose, handle, sort_keys=False)


def run_docker_compose(compose_path: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    down_cmd = ["docker", "compose", "-f", str(compose_path), "down", "--remove-orphans"]
    subprocess.run(
        down_cmd,
        cwd=str(DOCKER_DIR),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        return run_command(
            ["docker", "compose", "-f", str(compose_path), "up", "--build"],
            cwd=DOCKER_DIR,
            log_path=log_path,
            env=env,
        )
    finally:
        subprocess.run(
            down_cmd,
            cwd=str(DOCKER_DIR),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )


def run_local_campaign(
    rounds: int,
    matrix: List[Dict[str, Any]],
    staging_dir: Path,
    output_dir: Path,
    ollama_base_url: str,
    continue_on_error: bool,
    pattern_run_id: int,
) -> int:
    staging_dir.mkdir(parents=True, exist_ok=True)
    index_csv_path = staging_dir / "index.csv"

    original_config_text = read_text_if_exists(LOCAL_CONFIG_PATH)
    original_adaptation_text = read_text_if_exists(LOCAL_ADAPTATION_PATH)

    if not DOCKER_ADAPTATION_PATH.exists():
        raise FileNotFoundError(f"Missing adaptation source: {DOCKER_ADAPTATION_PATH}")

    failures = 0
    try:
        shutil.copy2(DOCKER_ADAPTATION_PATH, LOCAL_ADAPTATION_PATH)
        for idx, item in enumerate(matrix, start=1):
            spec = item["spec"]
            repeat_idx = item["repeat"]
            label = item["label"]
            run_dir = item["run_dir"]

            print(f"[{idx}/{len(matrix)}] Running {label} ({spec.description})")
            reset_runtime_outputs(LOCAL_DIR)

            config = build_local_config(
                spec,
                rounds=rounds,
                repeat_idx=repeat_idx,
                ollama_base_url=ollama_base_url,
                simulation_type="Local",
            )
            write_json(LOCAL_CONFIG_PATH, config)

            env = dict(os.environ)
            env["AP4FED_ROUNDS_OVERRIDE"] = str(rounds)
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

            archived = archive_run_outputs(run_dir, LOCAL_DIR, LOCAL_CONFIG_PATH, config)
            status = "ok" if rc == 0 else f"failed({rc})"
            append_stage_index_row(
                index_csv_path,
                {
                    "Configuration": spec.runner_name,
                    "Adaptation": spec.adaptation,
                    "LLM": spec.llm,
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
            else:
                print(f"[{idx}/{len(matrix)}] OK: {label} ({duration_seconds}s)")

            refresh_exported_outputs(staging_dir, output_dir, pattern_run_id)

            if rc != 0 and not continue_on_error:
                break
    finally:
        restore_text_file(LOCAL_CONFIG_PATH, original_config_text)
        restore_text_file(LOCAL_ADAPTATION_PATH, original_adaptation_text)

    return failures


def run_docker_campaign(
    rounds: int,
    matrix: List[Dict[str, Any]],
    staging_dir: Path,
    output_dir: Path,
    ollama_base_url: str,
    continue_on_error: bool,
    pattern_run_id: int,
) -> int:
    staging_dir.mkdir(parents=True, exist_ok=True)
    index_csv_path = staging_dir / "index.csv"

    original_config_text = read_text_if_exists(DOCKER_CONFIG_PATH)
    failures = 0
    try:
        for idx, item in enumerate(matrix, start=1):
            spec = item["spec"]
            repeat_idx = item["repeat"]
            label = item["label"]
            run_dir = item["run_dir"]

            print(f"[{idx}/{len(matrix)}] Running {label} ({spec.description})")
            reset_runtime_outputs(DOCKER_DIR)

            config = build_local_config(
                spec,
                rounds=rounds,
                repeat_idx=repeat_idx,
                ollama_base_url=ollama_base_url,
                simulation_type="Docker",
            )
            write_json(DOCKER_CONFIG_PATH, config)

            env = dict(os.environ)
            env["COMPOSE_BAKE"] = "true"

            compose_path = DOCKER_DIR / f"docker-compose.generated.{label}.yml"
            build_docker_compose(config, compose_path)
            safe_copy(compose_path, run_dir / "docker-compose.generated.yml")

            log_path = run_dir / "flower.log"
            started_at = time.time()
            try:
                rc = run_docker_compose(compose_path, log_path, env=env)
            finally:
                compose_path.unlink(missing_ok=True)
            duration_seconds = f"{time.time() - started_at:.1f}"

            archived = archive_run_outputs(run_dir, DOCKER_DIR, DOCKER_CONFIG_PATH, config)
            status = "ok" if rc == 0 else f"failed({rc})"
            append_stage_index_row(
                index_csv_path,
                {
                    "Configuration": spec.runner_name,
                    "Adaptation": spec.adaptation,
                    "LLM": spec.llm,
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
            else:
                print(f"[{idx}/{len(matrix)}] OK: {label} ({duration_seconds}s)")

            refresh_exported_outputs(staging_dir, output_dir, pattern_run_id)

            if rc != 0 and not continue_on_error:
                break
    finally:
        restore_text_file(DOCKER_CONFIG_PATH, original_config_text)

    return failures


def read_stage_index(staging_dir: Path) -> pd.DataFrame:
    index_path = staging_dir / "index.csv"
    if not index_path.exists():
        return pd.DataFrame(columns=INDEX_FIELDS)
    df = pd.read_csv(index_path)
    if df.empty:
        return pd.DataFrame(columns=INDEX_FIELDS)
    return df


def safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_stage_to_experiments(staging_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    manifest_rows: List[Dict[str, str]] = []
    for spec in APPROACH_SPECS:
        folder = output_dir / spec.folder_name
        folder.mkdir(parents=True, exist_ok=True)

        rows = staging_df.loc[staging_df["Configuration"] == spec.runner_name].copy()
        if rows.empty:
            continue
        rows["Repeat"] = pd.to_numeric(rows["Repeat"], errors="coerce")
        rows.sort_values("Repeat", inplace=True)

        first_config_written = False
        for _, row in rows.iterrows():
            run_dir = Path(str(row["Output Dir"]))
            repeat_idx = int(row["Repeat"])
            status = str(row["Status"])
            csv_src = run_dir / "performance" / "FLwithAP_performance_metrics.csv"
            csv_dst = folder / f"r{repeat_idx}.csv"
            rationale_src = run_dir / "performance" / "FLwithAP_adaptation_rationales.csv"
            rationale_dst = folder / f"r{repeat_idx}_rationales.csv"
            config_src = run_dir / "config.initial.json"
            config_dst = folder / "config.json"

            if status.startswith("ok") and csv_src.exists():
                safe_copy(csv_src, csv_dst)
            if rationale_src.exists():
                safe_copy(rationale_src, rationale_dst)
            if not first_config_written and config_src.exists():
                safe_copy(config_src, config_dst)
                first_config_written = True

            manifest_rows.append(
                {
                    "Configuration": spec.runner_name,
                    "Display Name": spec.display_name,
                    "Repeat": str(repeat_idx),
                    "Status": status,
                    "Stage Run Dir": str(run_dir),
                    "Exported CSV": str(csv_dst) if csv_dst.exists() else "",
                    "Exported Rationale CSV": str(rationale_dst) if rationale_dst.exists() else "",
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "run_manifest.csv", index=False)
    return manifest_df


def pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lowered = [(re.sub(r"\s+", " ", str(col).strip().lower()), col) for col in df.columns]
    for candidate in candidates:
        pattern = candidate.lower()
        for lowered_name, original_name in lowered:
            if pattern in lowered_name:
                return original_name
    return None


def choose_f1_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    f1_cols = [col for col in cols if "f1" in str(col).lower()]
    val_f1_cols = [col for col in f1_cols if any(tok in str(col).lower() for tok in ("val", "valid", "test"))]
    if val_f1_cols:
        return val_f1_cols[0]
    if f1_cols:
        return f1_cols[0]
    acc_cols = [col for col in cols if "accuracy" in str(col).lower()]
    val_acc_cols = [col for col in acc_cols if any(tok in str(col).lower() for tok in ("val", "valid", "test"))]
    if val_acc_cols:
        return val_acc_cols[0]
    return acc_cols[0] if acc_cols else None


def to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("\u202f", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.str.replace(r"(sec\.?|s)$", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def last_valid_value(series: pd.Series) -> float:
    numeric = to_numeric_series(series).dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else math.nan


def last_non_empty(series: pd.Series) -> str:
    values = [str(value).strip() for value in series.dropna() if str(value).strip()]
    return values[-1] if values else ""


def aggregate_run_csv(csv_path: Path, approach_name: str, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    round_col = pick_col(df, ["FL Round", "Round"])
    if round_col is None:
        raise ValueError(f"Missing round column in {csv_path}")

    f1_col = choose_f1_col(df)
    total_time_col = pick_col(df, ["Total Time of FL Round", "Total Round Time", "Total Time"])
    training_time_col = pick_col(df, ["Training Time"])
    communication_time_col = pick_col(df, ["Communication Time"])
    agent_time_col = pick_col(df, ["Agent Time (s)", "Agent Time", "Agent Overhead", "Reasoning Overhead"])
    ap_list_col = pick_col(df, ["AP List"])

    df = df.copy()
    df[round_col] = pd.to_numeric(df[round_col], errors="coerce")
    df = df.dropna(subset=[round_col])
    df[round_col] = df[round_col].astype(int)
    df.sort_values(round_col, inplace=True)

    rows: List[Dict[str, object]] = []
    for round_idx, sub in df.groupby(round_col, sort=True):
        rows.append(
            {
                "approach": approach_name,
                "run": run_name,
                "round": int(round_idx),
                "F1": last_valid_value(sub[f1_col]) if f1_col else math.nan,
                "total_time": last_valid_value(sub[total_time_col]) if total_time_col else math.nan,
                "training_time": float(to_numeric_series(sub[training_time_col]).mean()) if training_time_col else math.nan,
                "communication_time": float(to_numeric_series(sub[communication_time_col]).mean()) if communication_time_col else math.nan,
                "agent_time": last_valid_value(sub[agent_time_col]) if agent_time_col else 0.0,
                "ap_list": last_non_empty(sub[ap_list_col]) if ap_list_col else "",
            }
        )
    return pd.DataFrame(rows)


def load_exported_runs(output_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for spec in APPROACH_SPECS:
        folder = output_dir / spec.folder_name
        if not folder.exists():
            continue
        csv_files = sorted(
            [path for path in folder.glob("r*.csv") if re.fullmatch(r"r\d+", path.stem)],
            key=lambda path: int(re.search(r"\d+", path.stem).group(0)),
        )
        for csv_path in csv_files:
            run_name = csv_path.stem
            frames.append(aggregate_run_csv(csv_path, spec.display_name, run_name))
    if not frames:
        raise ValueError(f"No exported r*.csv files found under {output_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["approach", "run", "round"], inplace=True)
    return df


def fmt_mean_std(mean_value: float, std_value: float, suffix: str = "") -> str:
    if math.isnan(mean_value):
        return ""
    if suffix:
        return f"{int(round(mean_value))} ± {int(round(std_value))} {suffix}"
    return f"{mean_value:.2f} ± {std_value:.2f}"


def build_summary_table(round_df: pd.DataFrame) -> pd.DataFrame:
    run_rows: List[Dict[str, object]] = []
    for (approach, run_name), sub in round_df.groupby(["approach", "run"], sort=True):
        sub = sub.sort_values("round")
        final_f1 = float(sub["F1"].dropna().iloc[-1]) if sub["F1"].dropna().size else math.nan
        total_fl_time = float(sub["total_time"].dropna().sum()) if sub["total_time"].dropna().size else math.nan
        total_agent_overhead = float(sub["agent_time"].dropna().sum()) if sub["agent_time"].dropna().size else math.nan
        run_rows.append(
            {
                "Configuration": approach,
                "run": run_name,
                "final_f1": final_f1,
                "total_fl_time": total_fl_time,
                "agent_overhead": total_agent_overhead,
            }
        )

    per_run_df = pd.DataFrame(run_rows)
    summary_rows: List[Dict[str, object]] = []
    for display_name in DISPLAY_ORDER:
        sub = per_run_df.loc[per_run_df["Configuration"] == display_name]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "Configuration": display_name,
                "Model Accuracy (mean ± std)": fmt_mean_std(
                    float(sub["final_f1"].mean()),
                    float(sub["final_f1"].std(ddof=1)) if len(sub) > 1 else 0.0,
                ),
                "Total FL Time (mean ± std)": fmt_mean_std(
                    float(sub["total_fl_time"].mean()),
                    float(sub["total_fl_time"].std(ddof=1)) if len(sub) > 1 else 0.0,
                    "s",
                ),
                "Agent Overhead (mean ± std)": fmt_mean_std(
                    float(sub["agent_overhead"].mean()),
                    float(sub["agent_overhead"].std(ddof=1)) if len(sub) > 1 else 0.0,
                    "s",
                ),
                "n_runs": int(len(sub)),
            }
        )
    return pd.DataFrame(summary_rows)


def dynamic_xticks(rounds: Iterable[int]) -> List[int]:
    ordered = sorted(set(int(r) for r in rounds))
    if not ordered:
        return []
    max_round = ordered[-1]
    if max_round <= 10:
        return ordered
    if max_round <= 30:
        step = 5
    elif max_round <= 100:
        step = 10
    else:
        step = max(10, max_round // 10)
    ticks = [ordered[0]]
    ticks.extend(r for r in ordered if r % step == 0 and r not in ticks)
    if ordered[-1] not in ticks:
        ticks.append(ordered[-1])
    return ticks


def build_round_aggregate(round_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        round_df.groupby(["approach", "round"], as_index=False)
        .agg(
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            F1_count=("F1", "count"),
            total_time_mean=("total_time", "mean"),
            total_time_std=("total_time", "std"),
            total_time_count=("total_time", "count"),
            training_time_mean=("training_time", "mean"),
            training_time_std=("training_time", "std"),
            training_time_count=("training_time", "count"),
            communication_time_mean=("communication_time", "mean"),
            communication_time_std=("communication_time", "std"),
            communication_time_count=("communication_time", "count"),
        )
    )
    return grouped


def draw_metric_curve(
    aggregate_df: pd.DataFrame,
    metric_prefix: str,
    ylabel: str,
    output_path: Path,
    show_legend: bool = False,
    y_formatter: Optional[FuncFormatter] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    present = [name for name in DISPLAY_ORDER if name in set(aggregate_df["approach"])]
    rounds = sorted(set(int(val) for val in aggregate_df["round"]))

    for approach_name in present:
        sub = aggregate_df.loc[aggregate_df["approach"] == approach_name].sort_values("round")
        x = sub["round"].to_numpy(dtype=int)
        mean_values = sub[f"{metric_prefix}_mean"].to_numpy(dtype=float)
        std_values = sub[f"{metric_prefix}_std"].fillna(0.0).to_numpy(dtype=float)
        count_values = sub[f"{metric_prefix}_count"].clip(lower=1).to_numpy(dtype=float)
        ci = std_values / np.sqrt(count_values)
        low = mean_values - ci
        high = mean_values + ci
        if metric_prefix == "F1":
            low = np.clip(low, 0.0, 1.0)
            high = np.clip(high, 0.0, 1.0)

        ax.plot(
            x,
            mean_values,
            linestyle=DISPLAY_LINESTYLES[approach_name],
            color=DISPLAY_COLORS[approach_name],
            linewidth=1.6,
            label=approach_name,
            zorder=3,
        )
        ax.fill_between(x, low, high, color=DISPLAY_COLORS[approach_name], alpha=0.15, lw=0, zorder=2)

    ax.set_xlabel("Federated Learning Round")
    ax.set_ylabel(ylabel)
    ax.set_xticks(dynamic_xticks(rounds))
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
    if show_legend:
        ax.legend(loc="best", frameon=True, ncol=2)
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def parse_ap_list(value: object) -> Optional[List[str]]:
    if pd.isna(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        tokens = [str(item).strip().upper() for item in value]
    else:
        tokens = re.findall(r"\bON\b|\bOFF\b", str(value).upper())
    if not tokens:
        return None
    if len(tokens) < 6:
        tokens.extend(["OFF"] * (6 - len(tokens)))
    return tokens[:6]


def build_pattern_states(round_df: pd.DataFrame, pattern_run: str) -> Dict[str, Dict[str, List[int]]]:
    states: Dict[str, Dict[str, List[int]]] = {}
    for approach_name in DISPLAY_ORDER:
        sub = round_df.loc[(round_df["approach"] == approach_name) & (round_df["run"] == pattern_run)].copy()
        sub.sort_values("round", inplace=True)
        rounds = sub["round"].tolist()
        states[approach_name] = {"rounds": rounds, "CS": [], "MC": [], "HDH": []}
        for value in sub["ap_list"]:
            tokens = parse_ap_list(value)
            if tokens is None:
                cs = mc = hdh = 0
            else:
                cs = 1 if tokens[0] == "ON" else 0
                mc = 1 if tokens[2] == "ON" else 0
                hdh = 1 if tokens[5] == "ON" else 0
            states[approach_name]["CS"].append(cs)
            states[approach_name]["MC"].append(mc)
            states[approach_name]["HDH"].append(hdh)
    return states


def draw_pattern_activation(round_df: pd.DataFrame, output_path: Path, pattern_run: str) -> None:
    states = build_pattern_states(round_df, pattern_run)
    available_rounds = sorted(set(int(r) for r in round_df.loc[round_df["run"] == pattern_run, "round"]))
    if not available_rounds:
        return

    fig_height = max(4.5, len(available_rounds) * 0.12)
    fig, ax = plt.subplots(figsize=(12.0, fig_height))
    color_on = "#66c28c"
    color_off = "#e06b6b"
    edge_color = "#30363d"
    x_col_w = 0.03
    x_gap = 0.00
    group_gap = 0.05
    rect_height = 0.6

    x_lefts: Dict[Tuple[str, str], float] = {}
    for group_idx, approach_name in enumerate(DISPLAY_ORDER):
        x_base = group_idx * (3 * (x_col_w + x_gap) + group_gap)
        for pattern_idx, pattern_name in enumerate(PATTERN_NAMES):
            x_lefts[(approach_name, pattern_name)] = x_base + pattern_idx * (x_col_w + x_gap)

    all_lefts = [x_lefts[(approach_name, pattern_name)] for approach_name in DISPLAY_ORDER for pattern_name in PATTERN_NAMES]
    ax.set_xlim(min(all_lefts) - 0.02, max(all_lefts) + x_col_w + 0.05)
    ax.set_ylim(min(available_rounds) - 0.5, max(available_rounds) + 0.5)
    ax.set_ylabel("Federated Learning Round")
    ax.set_yticks(dynamic_xticks(available_rounds))

    for approach_name in DISPLAY_ORDER:
        approach_states = states.get(approach_name, {})
        rounds = approach_states.get("rounds", [])
        for pattern_name in PATTERN_NAMES:
            x_left = x_lefts[(approach_name, pattern_name)]
            values = approach_states.get(pattern_name, [])
            for round_idx, state in zip(rounds, values):
                ax.barh(
                    y=round_idx,
                    width=x_col_w,
                    left=x_left,
                    height=rect_height,
                    color=color_on if state else color_off,
                    edgecolor=edge_color,
                    linewidth=0.5,
                    hatch=None if state else "xx",
                    alpha=0.95,
                    zorder=3,
                )

    trans = ax.get_xaxis_transform()
    for approach_name in DISPLAY_ORDER:
        for pattern_name in PATTERN_NAMES:
            x_center = x_lefts[(approach_name, pattern_name)] + x_col_w / 2.0
            ax.text(x_center, -0.03, pattern_name, transform=trans, ha="center", va="top", fontsize=10, clip_on=False)

    for group_idx, approach_name in enumerate(DISPLAY_ORDER):
        x_base = group_idx * (3 * (x_col_w + x_gap) + group_gap)
        x_group_center = x_base + (x_col_w + x_gap) + x_col_w / 2.0
        ax.text(x_group_center, -0.10, approach_name, transform=trans, ha="center", va="top", fontsize=11, clip_on=False)

    ax.grid(False)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=color_on, edgecolor=edge_color, linewidth=0.5, label="Architectural Pattern Active"),
            mpatches.Patch(facecolor=color_off, edgecolor=edge_color, linewidth=0.5, hatch="xx", label="Architectural Pattern Not Active"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=True,
        framealpha=0.9,
    )
    plt.subplots_adjust(bottom=0.18)
    fig.savefig(output_path, bbox_inches="tight", dpi=600, facecolor="white")
    plt.close(fig)


def draw_score_boxplot(round_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = round_df.copy()
    plot_df["score"] = plot_df["F1"] / plot_df["total_time"].replace(0, np.nan)
    plot_df = plot_df.dropna(subset=["score"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(False, axis="x")

    data: List[np.ndarray] = []
    labels: List[str] = []
    for approach_name in DISPLAY_ORDER:
        subset = plot_df.loc[plot_df["approach"] == approach_name, "score"].to_numpy(dtype=float)
        if subset.size == 0:
            continue
        data.append(subset)
        labels.append(approach_name)

    if not data:
        plt.close(fig)
        return

    boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6, zorder=3)
    for patch, label in zip(boxplot["boxes"], labels):
        patch.set(
            facecolor=DISPLAY_COLORS.get(label, "#cccccc"),
            edgecolor="black",
            linewidth=0.8,
            hatch=DISPLAY_HATCHES.get(label, ""),
            zorder=3,
        )
    for median in boxplot["medians"]:
        x_data, y_data = median.get_xdata(), median.get_ydata()
        ax.plot(np.mean(x_data), np.mean(y_data), marker="^", markersize=7, color="black", zorder=10)
        median.set_alpha(0.0)

    ax.set_ylabel("Efficiency Score (Model Accuracy / Total Round Time)")
    ax.tick_params(axis="x", labelsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    legend_handles = [
        mpatches.Patch(
            facecolor=DISPLAY_COLORS[label],
            hatch=DISPLAY_HATCHES.get(label, ""),
            label=label,
            edgecolor="black",
        )
        for label in labels
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.54, 1.18), ncol=3, fontsize=9, frameon=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def k_formatter(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        scaled = value / 1000.0
        return f"{scaled:.1f}k" if scaled % 1 else f"{int(scaled)}k"
    return f"{int(value)}"


def generate_outputs(output_dir: Path, pattern_run_id: int) -> None:
    round_df = load_exported_runs(output_dir)
    round_df.to_csv(output_dir / "round_metrics.csv", index=False)

    summary_df = build_summary_table(round_df)
    summary_df.to_csv(output_dir / "LLM_summary_stats.csv", index=False)


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        print("--rounds must be >= 1", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    mode = args.mode.lower()
    output_dir = default_output_dir_for_mode(mode).resolve()
    staging_dir = Path(args.staging_dir).resolve() if args.staging_dir else default_staging_dir_for_mode(mode).resolve()
    ollama_base_url = args.ollama_base_url or default_ollama_url_for_mode(mode)

    if args.force:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if not args.skip_run and staging_dir.exists():
            shutil.rmtree(staging_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    sync_output_notebook(output_dir)
    if not args.skip_run:
        staging_dir.mkdir(parents=True, exist_ok=True)

    matrix, resume_plan = plan_resume_matrix(args.repeat, staging_dir, output_dir)

    planned = {
        "rounds": args.rounds,
        "repeat": args.repeat,
        "mode": mode,
        "output_dir": str(output_dir),
        "staging_dir": str(staging_dir),
        "ollama_base_url": ollama_base_url,
        "skip_run": bool(args.skip_run),
        "pattern_run": int(args.pattern_run),
        "scheduled_new_runs": len(matrix),
        "approaches": [spec.display_name for spec in APPROACH_SPECS],
        "resume": resume_plan,
    }
    print(json.dumps(planned, indent=2))
    if args.dry_run:
        return 0

    failures = 0
    if not args.skip_run:
        if not matrix:
            print("All requested runs are already present. No new local experiments will be launched.")
        else:
            if mode == "docker":
                failures = run_docker_campaign(
                    rounds=args.rounds,
                    matrix=matrix,
                    staging_dir=staging_dir,
                    output_dir=output_dir,
                    ollama_base_url=ollama_base_url,
                    continue_on_error=args.continue_on_error,
                    pattern_run_id=args.pattern_run,
                )
            else:
                failures = run_local_campaign(
                    rounds=args.rounds,
                    matrix=matrix,
                    staging_dir=staging_dir,
                    output_dir=output_dir,
                    ollama_base_url=ollama_base_url,
                    continue_on_error=args.continue_on_error,
                    pattern_run_id=args.pattern_run,
                )
    elif not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory does not exist: {staging_dir}")

    refresh_exported_outputs(staging_dir, output_dir, args.pattern_run)

    print(f"Experiments-style output ready in {output_dir}")
    if failures:
        print(f"Campaign completed with {failures} failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
