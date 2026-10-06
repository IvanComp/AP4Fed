#!/usr/bin/env python3
"""Run the AP4Fed pattern-stress campaign used by the ADEPT study.

The campaign runs only in Docker. The complete experiment matrix is scheduled
breadth-first by repetition: every configuration receives seed 1 before any
configuration receives seed 2, up to seed 10. Simulations stay serial so
concurrent workloads cannot contaminate performance measurements.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
LOCAL_DIR = ROOT / "Local"
DOCKER_DIR = ROOT / "Docker"
LOCAL_CONFIG_PATH = LOCAL_DIR / "configuration" / "config.json"
DOCKER_CONFIG_PATH = DOCKER_DIR / "configuration" / "config.json"
DOCKER_COMPOSE_PATH = DOCKER_DIR / "docker-compose.yml"
DOCKER_ADEPT_COMPOSE_PATH = DOCKER_DIR / "docker-compose.adept.yml"
DEFAULT_OUTPUT_DIR = ROOT / "pattern_stress_docker_results"
_DOCKER_IMAGES_READY = False

TASKS = (
    ("AG_NEWS", "MLP"),
    ("CIFAR-10", "CNN 16k"),
)
CLIENT_COUNTS = (4, 8, 10)
LOW_SPEC_PERCENTAGES = (25, 50, 75)
NON_IID_PERCENTAGES = (25, 50, 75)
REFERENCE_ALPHA = 0.5
DELAY_PERCENTAGES = (25, 50, 75)
MC_ENDPOINT_PERCENTAGES = (0, 100)
NOMINAL_STRESS_PERCENTAGE = 25
DELAY_MIN_SECONDS = 5
DELAY_MAX_SECONDS = 10
PARTITION_SEEDS = tuple(range(1, 11))
HOST_CPU_CAPACITY = 32
SERVER_CPUS = 2
LOW_SPEC_CPUS = 2
HIGH_SPEC_CPUS = 3
# AP4Fed serializes the states as Client Selector, Message Compressor, HDH.
CONFIGURATIONS = (
    "OFF,OFF,OFF",
    "ON,OFF,OFF",
    "OFF,OFF,ON",
    "OFF,ON,OFF",
)
CONFIGURATION_LABELS = {
    "OFF,OFF,OFF": "Baseline",
    "ON,OFF,OFF": "CS only",
    "OFF,ON,OFF": "MC only",
    "OFF,OFF,ON": "HDH only",
}


def stress_profiles(configuration: str) -> tuple[tuple[int, int, float, int], ...]:
    """Return legacy profiles plus matched high-stress and MC endpoint controls."""
    nominal_high_percentage = 100 - NOMINAL_STRESS_PERCENTAGE
    if configuration == "OFF,OFF,OFF":
        return (
            (
                nominal_high_percentage,
                NOMINAL_STRESS_PERCENTAGE,
                REFERENCE_ALPHA,
                NOMINAL_STRESS_PERCENTAGE,
            ),
            (25, 25, REFERENCE_ALPHA, 75),
            (25, 75, REFERENCE_ALPHA, 25),
            (75, 75, REFERENCE_ALPHA, 75),
            (75, 25, REFERENCE_ALPHA, 0),
            (75, 25, REFERENCE_ALPHA, 100),
        )
    if configuration == "ON,OFF,OFF":  # Client Selector
        legacy = tuple(
            (
                100 - low_percentage,
                NOMINAL_STRESS_PERCENTAGE,
                REFERENCE_ALPHA,
                NOMINAL_STRESS_PERCENTAGE,
            )
            for low_percentage in LOW_SPEC_PERCENTAGES
        )
        return legacy + (
            (25, 25, REFERENCE_ALPHA, 75),
            (25, 75, REFERENCE_ALPHA, 25),
        )
    if configuration == "OFF,OFF,ON":  # HDH
        legacy = tuple(
            (
                nominal_high_percentage,
                percentage,
                REFERENCE_ALPHA,
                NOMINAL_STRESS_PERCENTAGE,
            )
            for percentage in NON_IID_PERCENTAGES
        )
        return legacy + (
            (25, 75, REFERENCE_ALPHA, 25),
            (75, 75, REFERENCE_ALPHA, 75),
        )
    if configuration == "OFF,ON,OFF":  # Message Compressor
        legacy = tuple(
            (
                nominal_high_percentage,
                NOMINAL_STRESS_PERCENTAGE,
                REFERENCE_ALPHA,
                percentage,
            )
            for percentage in DELAY_PERCENTAGES
        )
        return legacy + tuple(
            (nominal_high_percentage, 25, REFERENCE_ALPHA, percentage)
            for percentage in MC_ENDPOINT_PERCENTAGES
        ) + (
            (25, 25, REFERENCE_ALPHA, 75),
            (75, 75, REFERENCE_ALPHA, 75),
        )
    raise ValueError(f"Unsupported pattern configuration: {configuration}")

INDEX_FIELDS = (
    "Run ID",
    "Execution Mode",
    "Wave",
    "Repeat",
    "Dataset",
    "Model",
    "Clients",
    "Low-spec Percent",
    "Realized Low-spec Percent",
    "High-spec Percent",
    "Realized High-spec Percent",
    "High-spec Clients",
    "Low-spec Clients",
    "Non-IID Percent",
    "Realized Non-IID Percent",
    "Non-IID Clients",
    "IID Clients",
    "Dirichlet Alpha",
    "Delay Percent",
    "Delayed Clients",
    "Realized Delay Percent",
    "Configuration",
    "Configuration Label",
    "Client Selector",
    "Message Compressor",
    "HDH",
    "Partition Seed",
    "Status",
    "Duration Seconds",
    "Output Dir",
    "ML Summary CSV",
    "Hostname",
    "Git Commit",
)

@dataclass(frozen=True)
class RunSpec:
    repeat: int
    dataset: str
    model: str
    client_count: int
    high_spec_percentage: int
    non_iid_percentage: int
    alpha: float
    delay_percentage: int
    configuration: str
    execution_mode: str = "Docker"

    @property
    def mode(self) -> str:
        normalized = self.execution_mode.strip().title()
        if normalized not in {"Local", "Docker"}:
            raise ValueError(f"Invalid execution mode: {self.execution_mode}")
        return normalized

    @property
    def states(self) -> tuple[str, str, str]:
        values = tuple(self.configuration.split(","))
        if len(values) != 3 or any(value not in {"ON", "OFF"} for value in values):
            raise ValueError(f"Invalid configuration: {self.configuration}")
        return values  # type: ignore[return-value]

    @property
    def low_spec_percentage(self) -> int:
        return 100 - self.high_spec_percentage

    @property
    def run_id(self) -> str:
        dataset_slug = sanitize_name(self.dataset.lower())
        model_slug = sanitize_name(self.model.lower())
        config_slug = self.configuration.lower().replace(",", "_")
        alpha_slug = str(self.alpha).replace(".", "p")
        return (
            f"adept__{self.mode.lower()}__{dataset_slug}__{model_slug}"
            f"__n{self.client_count}__l{self.low_spec_percentage}"
            f"__h{self.high_spec_percentage}"
            f"__ni{self.non_iid_percentage}__a{alpha_slug}"
            f"__d{self.delay_percentage}__{config_slug}__r{self.repeat:02d}"
        )


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def read_results_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()

    # FLwithAP_MLdata.csv is normally written with semicolon-separated fields
    # and decimal commas.  Looking only at the number of columns after a
    # default read is unsafe because the AP List header and values themselves
    # contain commas, making a semicolon CSV appear to have multiple columns.
    if header.count(";") > header.count(","):
        return pd.read_csv(path, sep=";", decimal=",")
    return pd.read_csv(path)


def split_pattern_column(
    frame: pd.DataFrame, fallback_config_id: str | None = None
) -> pd.DataFrame:
    result = frame.copy()
    if "config_id" in result.columns:
        return result

    ap_column = next((column for column in result.columns if str(column).startswith("AP List")), None)
    if ap_column is not None:
        states = (
            result[ap_column]
            .astype(str)
            .str.replace("{", "", regex=False)
            .str.replace("}", "", regex=False)
            .str.split(",", expand=True)
        )
        if states.shape[1] != 3:
            raise ValueError(f"Expected three pattern states in column '{ap_column}'")
        result["client_selector_pattern"] = states[0].str.strip()
        result["message_compressor_pattern"] = states[1].str.strip()
        result["hdh_pattern"] = states[2].str.strip()
        result = result.drop(columns=[ap_column])
    elif fallback_config_id:
        fallback_states = tuple(state.strip() for state in fallback_config_id.split(","))
        if len(fallback_states) != 3:
            raise ValueError(f"Invalid fallback pattern configuration: {fallback_config_id}")
        result["client_selector_pattern"] = fallback_states[0]
        result["message_compressor_pattern"] = fallback_states[1]
        result["hdh_pattern"] = fallback_states[2]
    else:
        raise ValueError("Results CSV has neither 'config_id' nor an 'AP List' column")

    result["config_id"] = (
        result["client_selector_pattern"]
        + ","
        + result["message_compressor_pattern"]
        + ","
        + result["hdh_pattern"]
    )
    return result


def verified_successful_run_ids(index_path: Path) -> set[str]:
    """Return only runs whose index row and archived artifacts are complete."""
    if not index_path.exists():
        return set()
    frame = pd.read_csv(index_path, dtype=str).fillna("")
    required = {"Run ID", "Status", "Output Dir", "ML Summary CSV"}
    if not required.issubset(frame.columns):
        return set()
    latest = frame.drop_duplicates("Run ID", keep="last")
    completed: set[str] = set()
    for _, row in latest.iterrows():
        if row["Status"] != "ok":
            continue
        output_dir = Path(row["Output Dir"])
        summary_path = Path(row["ML Summary CSV"])
        if output_dir.is_dir() and (output_dir / "config.json").is_file() and summary_path.is_file():
            completed.add(str(row["Run ID"]))
    return completed


def build_plan(
    completed_run_ids: set[str],
    target_repeats: int,
    tasks: Iterable[tuple[str, str]] = TASKS,
    client_counts: Iterable[int] = CLIENT_COUNTS,
    configurations: Iterable[str] = CONFIGURATIONS,
    through_wave: int | None = None,
    only_wave: int | None = None,
    execution_mode: str = "Docker",
) -> list[RunSpec]:
    if target_repeats < 1:
        raise ValueError("target_repeats must be >= 1")
    if target_repeats > len(PARTITION_SEEDS):
        raise ValueError(f"target_repeats must be <= {len(PARTITION_SEEDS)}")
    if through_wave is not None and through_wave < 1:
        raise ValueError("through_wave must be >= 1")
    if only_wave is not None and not 1 <= only_wave <= target_repeats:
        raise ValueError(f"only_wave must be between 1 and {target_repeats}")
    if through_wave is not None and only_wave is not None:
        raise ValueError("through_wave and only_wave cannot be used together")

    last_wave = min(target_repeats, through_wave) if through_wave else target_repeats
    repeats = (only_wave,) if only_wave is not None else range(1, last_wave + 1)
    plan: list[RunSpec] = []
    # Complete the whole matrix for one seed before starting the next repetition.
    for repeat in repeats:
        for dataset, model in tasks:
            for client_count in client_counts:
                for configuration in configurations:
                    for (
                        high_spec_percentage,
                        non_iid_percentage,
                        alpha,
                        delay_percentage,
                    ) in stress_profiles(configuration):
                        spec = RunSpec(
                            repeat=repeat,
                            dataset=dataset,
                            model=model,
                            client_count=int(client_count),
                            high_spec_percentage=int(high_spec_percentage),
                            non_iid_percentage=int(non_iid_percentage),
                            alpha=float(alpha),
                            delay_percentage=int(delay_percentage),
                            configuration=configuration,
                            execution_mode=execution_mode,
                        )
                        if spec.run_id not in completed_run_ids:
                            plan.append(spec)
    return plan


def build_partition_seed(spec: RunSpec, rounds: int) -> int:
    del rounds
    if not 1 <= spec.repeat <= len(PARTITION_SEEDS):
        raise ValueError(f"repeat must be between 1 and {len(PARTITION_SEEDS)}")
    return PARTITION_SEEDS[spec.repeat - 1]


def configure_resource_profile(
    host_cpus: int,
    server_cpus: int,
    low_spec_cpus: int,
    high_spec_cpus: int,
) -> None:
    global HOST_CPU_CAPACITY, SERVER_CPUS, LOW_SPEC_CPUS, HIGH_SPEC_CPUS
    if min(host_cpus, server_cpus, low_spec_cpus, high_spec_cpus) < 1:
        raise ValueError("All CPU profile values must be positive integers")
    if high_spec_cpus <= low_spec_cpus:
        raise ValueError("high-spec CPUs must be greater than low-spec CPUs")
    max_clients = max(CLIENT_COUNTS)
    max_high = _percentage_count(max_clients, 100 - min(LOW_SPEC_PERCENTAGES))
    required = (
        server_cpus
        + max_high * high_spec_cpus
        + (max_clients - max_high) * low_spec_cpus
    )
    if required > host_cpus:
        raise ValueError(
            f"CPU profile requires {required} cores in the largest configuration; "
            f"host capacity is {host_cpus}"
        )
    HOST_CPU_CAPACITY = host_cpus
    SERVER_CPUS = server_cpus
    LOW_SPEC_CPUS = low_spec_cpus
    HIGH_SPEC_CPUS = high_spec_cpus


def build_patterns(spec: RunSpec) -> dict[str, dict[str, Any]]:
    selector, compressor, hdh = spec.states
    return {
        "client_registry": {"enabled": True, "params": {}},
        "client_selector": {
            "enabled": selector == "ON",
            "params": {
                "selection_strategy": "Resource-Based",
                "selection_criteria": "CPU",
                "selection_value": LOW_SPEC_CPUS,
            },
        },
        "client_cluster": {"enabled": False, "params": {}},
        "message_compressor": {"enabled": compressor == "ON", "params": {}},
        "model_co-versioning_registry": {"enabled": False, "params": {}},
        "multi-task_model_trainer": {"enabled": False, "params": {}},
        "heterogeneous_data_handler": {"enabled": hdh == "ON", "params": {}},
    }


def _percentage_count(client_count: int, percentage: int) -> int:
    if client_count < 2:
        raise ValueError("client_count must be >= 2")
    if not 0 <= percentage <= 100:
        raise ValueError(f"Unsupported client percentage: {percentage}")
    if percentage == 0:
        return 0
    if percentage == 100:
        return client_count
    # Python's tie-to-even rounding keeps the N=10 endpoints complementary:
    # 25% -> 2 clients, 75% -> 8 clients.
    return max(1, min(client_count - 1, round(client_count * percentage / 100.0)))


def _selected_client_ids(
    client_ids: list[int], percentage: int, seed: int, salt: int
) -> set[int]:
    count = _percentage_count(len(client_ids), percentage)
    ordered_ids = list(client_ids)
    random.Random(seed ^ salt).shuffle(ordered_ids)
    return set(ordered_ids[:count])


def build_config(spec: RunSpec, rounds: int) -> dict[str, Any]:
    partition_seed = build_partition_seed(spec, rounds)
    client_ids = list(range(1, spec.client_count + 1))
    high_ids = _selected_client_ids(
        client_ids, spec.high_spec_percentage, partition_seed, 0xC51EC7
    )
    non_iid_ids = _selected_client_ids(
        client_ids, spec.non_iid_percentage, partition_seed, 0xA1F4A
    )
    delayed_ids = _selected_client_ids(
        client_ids, spec.delay_percentage, partition_seed, 0xDE1A7
    )
    high_count = len(high_ids)
    non_iid_count = len(non_iid_ids)
    delayed_count = len(delayed_ids)
    clients = []
    for client_id in client_ids:
        delayed = client_id in delayed_ids
        non_iid = client_id in non_iid_ids
        clients.append(
            {
                "client_id": client_id,
                "cpu": HIGH_SPEC_CPUS if client_id in high_ids else LOW_SPEC_CPUS,
                "ram": 4,
                "dataset": spec.dataset,
                "data_distribution_type": "non-IID" if non_iid else "IID",
                "non_iid_alpha": spec.alpha if non_iid else 1.0,
                "data_persistence_type": "Same Data",
                "delay_combobox": "Yes" if delayed else "No",
                "delay_min_seconds": DELAY_MIN_SECONDS if delayed else 0,
                "delay_max_seconds": DELAY_MAX_SECONDS if delayed else 0,
                "model": spec.model,
                "epochs": 1,
            }
        )
    return {
        "simulation_type": spec.mode,
        "rounds": int(rounds),
        "clients": len(clients),
        "clients_per_round": len(clients),
        "dataset": spec.dataset,
        "adaptation": "None",
        "LLM": "llama3.2:3b",
        "partition_seed": partition_seed,
        "campaign_metadata": {
            "configuration_label": CONFIGURATION_LABELS[spec.configuration],
            "low_spec_percentage_requested": spec.low_spec_percentage,
            "realized_low_spec_percentage": 100.0 * (spec.client_count - high_count) / spec.client_count,
            "high_spec_percentage_requested": spec.high_spec_percentage,
            "realized_high_spec_percentage": 100.0 * high_count / spec.client_count,
            "high_spec_clients": high_count,
            "low_spec_clients": spec.client_count - high_count,
            "non_iid_percentage_requested": spec.non_iid_percentage,
            "realized_non_iid_percentage": 100.0 * non_iid_count / spec.client_count,
            "non_iid_clients": non_iid_count,
            "iid_clients": spec.client_count - non_iid_count,
            "dirichlet_alpha": spec.alpha,
            "delay_percentage_requested": spec.delay_percentage,
            "delayed_clients": delayed_count,
            "realized_delay_percentage": 100.0 * delayed_count / spec.client_count,
        },
        "patterns": build_patterns(spec),
        "client_generation_mode": "manual",
        "client_profiles": [],
        "client_details": clients,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def reset_runtime_state(work_dir: Path, mode: str) -> None:
    for folder_name in ("performance", "performance_MLdata", "logs"):
        folder = work_dir / folder_name
        if not folder.exists():
            continue
        for entry in folder.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    if mode == "Local":
        (work_dir / ".client_idx").write_text("0", encoding="utf-8")
        (work_dir / ".cpu_pool_state.json").write_text('{"allocations": {}}\n', encoding="utf-8")


def ensure_datasets_available(tasks: Iterable[tuple[str, str]]) -> None:
    """Prepare each selected dataset once and share the cache with Docker."""
    selected = {dataset for dataset, _ in tasks}
    data_root = LOCAL_DIR / "data"
    if "AG_NEWS" in selected:
        agnews_root = data_root / "ag_news_csv"
        agnews_root.mkdir(parents=True, exist_ok=True)
        sources = {
            "train.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
            "test.csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
        }
        for filename, url in sources.items():
            destination = agnews_root / filename
            if not destination.exists():
                print(f"Downloading AG_NEWS {filename} ...")
                urllib.request.urlretrieve(url, destination)
    if "CIFAR-10" in selected:
        from torchvision.datasets import CIFAR10

        print(f"Checking CIFAR-10 dataset cache in {data_root} ...")
        CIFAR10(root=str(data_root), train=True, download=True)
        CIFAR10(root=str(data_root), train=False, download=True)
    shutil.copytree(data_root, DOCKER_DIR / "data", dirs_exist_ok=True)
    print("Selected dataset caches are ready for Docker.")


def ensure_datasets_available_singularity(image_path: Path) -> None:
    singularity = resolve_singularity_command()
    if singularity is None:
        raise RuntimeError("Neither singularity nor apptainer is available in PATH")
    prefetch_code = (
        "from taskA import AGNewsDataset, CIFAR10; "
        "AGNewsDataset('train'); AGNewsDataset('test'); "
        "CIFAR10('./data', train=True, download=True); "
        "CIFAR10('./data', train=False, download=True)"
    )
    subprocess.run(
        [
            singularity,
            "exec",
            "--cleanenv",
            "--bind",
            f"{DOCKER_DIR}:/app",
            "--pwd",
            "/app",
            str(image_path.resolve()),
            "python",
            "-c",
            prefetch_code,
        ],
        cwd=DOCKER_DIR,
        check=True,
    )
    print("Selected dataset caches are ready for Singularity.")


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def archive_outputs(run_dir: Path, config: dict[str, Any], work_dir: Path) -> Path | None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    copy_if_exists(work_dir / "performance", run_dir / "performance")
    copy_if_exists(work_dir / "performance_MLdata", run_dir / "performance_MLdata")
    copy_if_exists(work_dir / "logs", run_dir / "logs")
    summary = run_dir / "performance_MLdata" / "FLwithAP_MLdata.csv"
    return summary if summary.exists() else None


def append_index_row(index_path: Path, row: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})


def regenerate_combined_dataset(
    phase_indexes: Iterable[tuple[str, Path]],
    destination: Path,
) -> int:
    frames: list[pd.DataFrame] = []
    for mode, index_path in phase_indexes:
        if not index_path.exists():
            continue
        index = pd.read_csv(index_path, dtype=str).fillna("")
        successful = index[index["Status"] == "ok"].drop_duplicates("Run ID", keep="last")
        for _, record in successful.iterrows():
            summary_path = Path(record["ML Summary CSV"])
            if not summary_path.exists():
                continue
            try:
                frame = split_pattern_column(
                    read_results_csv(summary_path),
                    fallback_config_id=record.get("Configuration", ""),
                )
            except Exception as exc:
                print(
                    f"WARNING: unable to merge summary for {record['Run ID']}: {exc}",
                    file=sys.stderr,
                )
                continue
            frame["repeat"] = int(record["Repeat"])
            frame["run_id"] = record["Run ID"]
            frame["execution_mode"] = mode
            frame["dataset"] = record.get("Dataset", "")
            frame["clients"] = int(record.get("Clients", 0) or 0)
            frame["low_spec_percentage"] = int(record.get("Low-spec Percent", 0) or 0)
            frame["realized_low_spec_percentage"] = float(
                record.get("Realized Low-spec Percent", 0) or 0
            )
            frame["high_spec_percentage"] = int(record.get("High-spec Percent", 0) or 0)
            frame["realized_high_spec_percentage"] = float(
                record.get("Realized High-spec Percent", 0) or 0
            )
            frame["non_iid_percentage"] = int(record.get("Non-IID Percent", 0) or 0)
            frame["realized_non_iid_percentage"] = float(
                record.get("Realized Non-IID Percent", 0) or 0
            )
            frame["non_iid_clients"] = int(record.get("Non-IID Clients", 0) or 0)
            frame["dirichlet_alpha"] = float(record.get("Dirichlet Alpha", 0) or 0)
            frame["delay_percentage"] = int(record.get("Delay Percent", 0) or 0)
            frame["delayed_clients"] = int(record.get("Delayed Clients", 0) or 0)
            frame["realized_delay_percentage"] = float(record.get("Realized Delay Percent", 0) or 0)
            frame["hostname"] = record.get("Hostname", "")
            frame["git_commit"] = record.get("Git Commit", "")
            frame["partition_seed"] = int(record["Partition Seed"])
            frames.append(frame)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        pd.DataFrame().to_csv(destination, index=False)
        return 0
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(destination, index=False)
    return len(combined)


def regenerate_client_round_dataset(
    phase_indexes: Iterable[tuple[str, Path]],
    destination: Path,
) -> int:
    """Create one row per configured client and FL round for every completed run."""
    frames: list[pd.DataFrame] = []
    for mode, index_path in phase_indexes:
        if not index_path.exists():
            continue
        index = pd.read_csv(index_path, dtype=str).fillna("")
        successful = index[index["Status"] == "ok"].drop_duplicates("Run ID", keep="last")
        for _, record in successful.iterrows():
            output_dir = Path(record["Output Dir"])
            metrics_path = output_dir / "performance" / "FLwithAP_performance_metrics.csv"
            config_path = output_dir / "config.json"
            if not metrics_path.is_file() or not config_path.is_file():
                print(
                    f"WARNING: unable to merge client-round metrics for {record['Run ID']}: "
                    "missing performance CSV or config.json",
                    file=sys.stderr,
                )
                continue

            try:
                metrics = pd.read_csv(metrics_path)
                config = json.loads(config_path.read_text(encoding="utf-8"))
                client_details = config.get("client_details", [])
                rounds = int(config.get("rounds", 0))
                if not client_details or rounds < 1:
                    raise ValueError("config.json has no clients or rounds")

                metrics["client_number"] = pd.to_numeric(
                    metrics["Client ID"].astype(str).str.extract(r"(\d+)")[0],
                    errors="coerce",
                ).astype("Int64")
                metrics["FL Round"] = pd.to_numeric(
                    metrics["FL Round"], errors="coerce"
                ).astype("Int64")
                metrics = metrics.dropna(subset=["client_number", "FL Round"])
                metrics = metrics.drop_duplicates(
                    ["client_number", "FL Round"], keep="last"
                )

                grid_rows = []
                for client in client_details:
                    client_number = int(client["client_id"])
                    for round_number in range(1, rounds + 1):
                        grid_rows.append(
                            {
                                "client_number": client_number,
                                "Client ID": f"Client {client_number}",
                                "FL Round": round_number,
                                "configured_cpu": client.get("cpu"),
                                "configured_ram_gb": client.get("ram"),
                                "configured_data_distribution": client.get(
                                    "data_distribution_type"
                                ),
                                "configured_data_persistence": client.get(
                                    "data_persistence_type"
                                ),
                                "configured_dirichlet_alpha": client.get(
                                    "non_iid_alpha"
                                ),
                                "configured_delay": client.get("delay_combobox") == "Yes",
                                "configured_delay_min_seconds": client.get(
                                    "delay_min_seconds"
                                ),
                                "configured_delay_max_seconds": client.get(
                                    "delay_max_seconds"
                                ),
                            }
                        )
                grid = pd.DataFrame(grid_rows)
                metrics = metrics.drop(columns=["Client ID"])
                frame = grid.merge(
                    metrics,
                    on=["client_number", "FL Round"],
                    how="left",
                    indicator=True,
                    validate="one_to_one",
                )
                frame["participated"] = frame.pop("_merge").eq("both")
                frame = frame.sort_values(
                    ["FL Round", "client_number"], kind="stable"
                ).reset_index(drop=True)
                frame = frame.drop(columns=["client_number"])
            except Exception as exc:
                print(
                    f"WARNING: unable to merge client-round metrics for {record['Run ID']}: {exc}",
                    file=sys.stderr,
                )
                continue

            configuration = record.get("Configuration", "")
            pattern_states = tuple(state.strip() for state in configuration.split(","))
            if len(pattern_states) != 3:
                print(
                    f"WARNING: invalid configuration for {record['Run ID']}: {configuration}",
                    file=sys.stderr,
                )
                continue
            frame["client_selector_pattern"] = pattern_states[0]
            frame["message_compressor_pattern"] = pattern_states[1]
            frame["hdh_pattern"] = pattern_states[2]
            frame["config_id"] = configuration
            frame["repeat"] = int(record["Repeat"])
            frame["run_id"] = record["Run ID"]
            frame["execution_mode"] = mode
            frame["dataset"] = record.get("Dataset", "")
            frame["model"] = record.get("Model", "")
            frame["clients"] = int(record.get("Clients", 0) or 0)
            frame["low_spec_percentage"] = int(record.get("Low-spec Percent", 0) or 0)
            frame["realized_low_spec_percentage"] = float(
                record.get("Realized Low-spec Percent", 0) or 0
            )
            frame["high_spec_percentage"] = int(record.get("High-spec Percent", 0) or 0)
            frame["realized_high_spec_percentage"] = float(
                record.get("Realized High-spec Percent", 0) or 0
            )
            frame["non_iid_percentage"] = int(record.get("Non-IID Percent", 0) or 0)
            frame["realized_non_iid_percentage"] = float(
                record.get("Realized Non-IID Percent", 0) or 0
            )
            frame["dirichlet_alpha"] = float(record.get("Dirichlet Alpha", 0) or 0)
            frame["delay_percentage"] = int(record.get("Delay Percent", 0) or 0)
            frame["realized_delay_percentage"] = float(
                record.get("Realized Delay Percent", 0) or 0
            )
            frame["hostname"] = record.get("Hostname", "")
            frame["git_commit"] = record.get("Git Commit", "")
            frame["partition_seed"] = int(record["Partition Seed"])
            frames.append(frame)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        pd.DataFrame().to_csv(destination, index=False)
        return 0
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        ["run_id", "FL Round", "Client ID"], keep="last"
    )
    combined.to_csv(destination, index=False)
    return len(combined)


def build_docker_compose(
    config: dict[str, Any],
    source: Path = DOCKER_COMPOSE_PATH,
    destination: Path = DOCKER_ADEPT_COMPOSE_PATH,
) -> Path:
    import yaml

    compose = yaml.safe_load(source.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    server = copy.deepcopy(services.get("server"))
    client_template = services.get("client")
    if not server or not client_template:
        raise ValueError(f"Missing server/client service in {source}")

    server.pop("deploy", None)
    server["cpus"] = SERVER_CPUS
    server["cpuset"] = ",".join(str(cpu_id) for cpu_id in range(SERVER_CPUS))
    server["mem_limit"] = "8g"
    server.setdefault("environment", {})["NUM_ROUNDS"] = str(config["rounds"])
    generated_services = {"server": server}
    next_cpu_id = SERVER_CPUS
    for detail in config["client_details"]:
        client_id = int(detail["client_id"])
        cpu = int(detail["cpu"])
        ram = int(detail["ram"])
        service = copy.deepcopy(client_template)
        service.pop("deploy", None)
        service["container_name"] = f"Client{client_id}"
        service["cpus"] = cpu
        assigned_cpu_ids = list(range(next_cpu_id, next_cpu_id + cpu))
        next_cpu_id += cpu
        if next_cpu_id > HOST_CPU_CAPACITY:
            raise ValueError(
                f"Configuration requires {next_cpu_id} CPU cores, "
                f"but the host capacity is {HOST_CPU_CAPACITY}"
            )
        cpu_set = ",".join(str(cpu_id) for cpu_id in assigned_cpu_ids)
        service["cpuset"] = cpu_set
        service["mem_limit"] = f"{ram}g"
        environment = service.setdefault("environment", {})
        environment.update(
            {
                "CLIENT_ID": str(client_id),
                "NUM_CPUS": str(cpu),
                "NUM_RAM": str(ram),
                "NUM_ROUNDS": str(config["rounds"]),
                "CPUSET_CPUS": cpu_set,
            }
        )
        generated_services[f"client{client_id}"] = service

    compose["services"] = generated_services
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")
    return destination


def resolve_docker_compose_command() -> list[str] | None:
    docker = shutil.which("docker")
    if docker:
        result = subprocess.run(
            [docker, "compose", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return [docker, "compose"]
    legacy = shutil.which("docker-compose")
    return [legacy] if legacy else None


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def lock_campaign_to_machine(output_dir: Path, machine_id: str | None = None) -> tuple[str, str]:
    if platform.system() != "Linux":
        raise RuntimeError("The ADEPT campaign is locked to the Linux workstation and cannot run on this computer.")

    hostname = socket.gethostname()
    campaign_machine_id = machine_id or hostname
    commit = git_commit()
    manifest_path = output_dir / "campaign_machine.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_machine_id = str(manifest.get("machine_id", manifest.get("hostname", "")))
        if recorded_machine_id and recorded_machine_id != campaign_machine_id:
            raise RuntimeError(
                f"Campaign belongs to machine '{recorded_machine_id}', "
                f"not current machine '{campaign_machine_id}'."
            )
    else:
        write_json(
            manifest_path,
            {
                "hostname": hostname,
                "machine_id": campaign_machine_id,
                "platform": platform.platform(),
                "git_commit_at_start": commit,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "phase_order": ["Docker"],
            },
        )
    return hostname, commit


def phase_index_path(output_dir: Path, mode: str) -> Path:
    return output_dir / mode.lower() / "index.csv"


def print_plan(mode: str, plan: list[RunSpec]) -> None:
    print(f"\n{mode} phase: {len(plan)} scheduled run(s)")
    for wave in sorted({spec.repeat for spec in plan}):
        wave_specs = [spec for spec in plan if spec.repeat == wave]
        task_counts = ", ".join(
            f"{dataset}/{model}: "
            f"{sum(spec.dataset == dataset and spec.model == model for spec in wave_specs)}"
            for dataset, model in TASKS
        )
        print(f"- Wave {wave} (seed {wave}): {len(wave_specs)} run(s) [{task_counts}]")
    if plan:
        print(f"Next run: {plan[0].run_id}")


def validate_complete_matrix(plan: list[RunSpec], target_repeats: int) -> None:
    expected_per_wave = len(TASKS) * len(CLIENT_COUNTS) * sum(
        len(stress_profiles(configuration)) for configuration in CONFIGURATIONS
    )
    expected_total = expected_per_wave * target_repeats
    run_ids = [spec.run_id for spec in plan]
    if len(plan) != expected_total:
        raise RuntimeError(
            f"Incomplete campaign matrix: found {len(plan)} runs, expected {expected_total}"
        )
    if len(set(run_ids)) != expected_total:
        raise RuntimeError("Campaign matrix contains duplicate run identifiers")
    for repeat in range(1, target_repeats + 1):
        wave = [spec for spec in plan if spec.repeat == repeat]
        if len(wave) != expected_per_wave:
            raise RuntimeError(
                f"Wave {repeat} contains {len(wave)} runs, expected {expected_per_wave}"
            )
        if any(build_partition_seed(spec, 20) != repeat for spec in wave):
            raise RuntimeError(f"Wave {repeat} is not consistently tied to seed {repeat}")


def run_local_process(config: dict[str, Any], log_path: Path, rounds: int) -> int:
    env = dict(os.environ)
    env["AP4FED_ROUNDS_OVERRIDE"] = str(rounds)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            ["flower-simulation", "--app", ".", "--num-supernodes", str(config["clients"])],
            cwd=LOCAL_DIR,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


def run_docker_process(config: dict[str, Any], log_path: Path, compose_project: str) -> int:
    global _DOCKER_IMAGES_READY
    compose_command = resolve_docker_compose_command()
    if compose_command is None:
        raise RuntimeError("Docker Compose is not installed or is not available in PATH")

    compose_path = build_docker_compose(config)
    base_command = compose_command + ["-p", compose_project, "-f", str(compose_path)]
    env = dict(os.environ)
    env["COMPOSE_BAKE"] = "true"
    env["NUM_ROUNDS"] = str(config["rounds"])

    subprocess.run(
        base_command + ["down", "--volumes", "--remove-orphans"],
        cwd=DOCKER_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            if not _DOCKER_IMAGES_READY:
                build_process = subprocess.run(
                    base_command + ["build"],
                    cwd=DOCKER_DIR,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if build_process.returncode != 0:
                    return int(build_process.returncode)
                prefetch_code = (
                    "from taskA import AGNewsDataset, CIFAR10; "
                    "AGNewsDataset('train'); AGNewsDataset('test'); "
                    "CIFAR10('./data', train=True, download=True); "
                    "CIFAR10('./data', train=False, download=True)"
                )
                prefetch_process = subprocess.run(
                    base_command
                    + [
                        "run",
                        "--rm",
                        "--no-deps",
                        "client1",
                        "python",
                        "-c",
                        prefetch_code,
                    ],
                    cwd=DOCKER_DIR,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if prefetch_process.returncode != 0:
                    return int(prefetch_process.returncode)
                _DOCKER_IMAGES_READY = True
            process = subprocess.run(
                base_command
                + [
                    "up",
                    "--no-build",
                    "--remove-orphans",
                    "--abort-on-container-exit",
                    "--exit-code-from",
                    "server",
                ],
                cwd=DOCKER_DIR,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return int(process.returncode)
    finally:
        with log_path.open("a", encoding="utf-8") as log_handle:
            subprocess.run(
                base_command + ["down", "--volumes", "--remove-orphans"],
                cwd=DOCKER_DIR,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )


def resolve_singularity_command() -> str | None:
    return shutil.which("singularity") or shutil.which("apptainer")


def build_singularity_step_command(
    singularity: str,
    image_path: Path,
    cpus: int,
    memory_gb: int,
    environment: dict[str, str],
    program: str,
) -> list[str]:
    command = [
        "srun",
        "--exclusive",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem={memory_gb}G",
        "--cpu-bind=cores",
        singularity,
        "exec",
        "--cleanenv",
        "--bind",
        f"{DOCKER_DIR}:/app",
        "--pwd",
        "/app",
    ]
    for key, value in environment.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend([str(image_path), "python", program])
    return command


def _terminate_processes(
    processes: list[subprocess.Popen], graceful_timeout: int = 30
) -> None:
    # Flower asks every client to disconnect after the server completes.  Give
    # the corresponding srun steps time to exit normally before signalling
    # them; terminating the launchers immediately can make Slurm abort the
    # steps and prevent the next experiment from starting cleanly.
    graceful_deadline = time.time() + graceful_timeout
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, graceful_deadline - time.time()))
            except subprocess.TimeoutExpired:
                break

    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 15
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def _wait_for_server(process: subprocess.Popen, port: int, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def run_singularity_process(
    config: dict[str, Any], log_path: Path, image_path: Path
) -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The Singularity backend must run inside a Slurm allocation")
    singularity = resolve_singularity_command()
    if singularity is None:
        raise RuntimeError("Neither singularity nor apptainer is available in PATH")
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Singularity image not found: {image_path}")

    processes: list[subprocess.Popen] = []
    slurm_job_id = int(os.environ["SLURM_JOB_ID"].split("_")[0])
    server_port = 20000 + (slurm_job_id % 20000)
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            server_command = build_singularity_step_command(
                singularity,
                image_path,
                cpus=SERVER_CPUS,
                memory_gb=8,
                environment={
                    "NUM_ROUNDS": str(config["rounds"]),
                    "SERVER_PORT": str(server_port),
                    "PYTHONUNBUFFERED": "1",
                },
                program="server.py",
            )
            server = subprocess.Popen(
                server_command,
                cwd=DOCKER_DIR,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(server)
            if not _wait_for_server(server, server_port):
                return int(server.poll() or 1)

            clients: list[subprocess.Popen] = []
            for detail in config["client_details"]:
                client_command = build_singularity_step_command(
                    singularity,
                    image_path,
                    cpus=int(detail["cpu"]),
                    memory_gb=int(detail["ram"]),
                    environment={
                        "CLIENT_ID": str(detail["client_id"]),
                        "NUM_CPUS": str(detail["cpu"]),
                        "NUM_RAM": str(detail["ram"]),
                        "NUM_ROUNDS": str(config["rounds"]),
                        "SERVER_ADDRESS": f"127.0.0.1:{server_port}",
                        "PYTHONUNBUFFERED": "1",
                    },
                    program="client.py",
                )
                client = subprocess.Popen(
                    client_command,
                    cwd=DOCKER_DIR,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                clients.append(client)
                processes.append(client)

            while server.poll() is None:
                failed_client = next(
                    (client for client in clients if client.poll() not in (None, 0)),
                    None,
                )
                if failed_client is not None:
                    return int(failed_client.returncode or 1)
                time.sleep(1)
            return int(server.returncode or 0)
        finally:
            _terminate_processes(processes)


def execute_phase(
    mode: str,
    plan: list[RunSpec],
    output_dir: Path,
    phase_indexes: tuple[tuple[str, Path], ...],
    combined_path: Path,
    rounds: int,
    hostname: str,
    commit: str,
    continue_on_error: bool,
    compose_project: str,
    container_runtime: str = "docker",
    singularity_image: Path | None = None,
) -> int:
    if not plan:
        print(f"\n{mode} phase already complete.")
        return 0

    work_dir = LOCAL_DIR if mode == "Local" else DOCKER_DIR
    config_path = LOCAL_CONFIG_PATH if mode == "Local" else DOCKER_CONFIG_PATH
    index_path = phase_index_path(output_dir, mode)
    failures = 0

    print(f"\nStarting {mode} phase on host {hostname}.")
    for position, spec in enumerate(plan, start=1):
        config = build_config(spec, rounds)
        run_dir = output_dir / mode.lower() / "runs" / spec.run_id
        if mode == "Local":
            log_name = "flower.log"
        elif container_runtime == "singularity":
            log_name = "singularity-slurm.log"
        else:
            log_name = "docker-compose.log"
        log_path = run_dir / log_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[{mode} {position}/{len(plan)}] Wave {spec.repeat}: "
            f"{spec.dataset}/{spec.model} / {CONFIGURATION_LABELS[spec.configuration]}"
        )

        reset_runtime_state(work_dir, mode)
        write_json(config_path, config)
        started = time.time()
        startup_error = ""
        interrupted = False
        try:
            if mode == "Local":
                return_code = run_local_process(config, log_path, rounds)
            elif container_runtime == "singularity":
                if singularity_image is None:
                    raise RuntimeError("A Singularity image is required")
                return_code = run_singularity_process(config, log_path, singularity_image)
            else:
                return_code = run_docker_process(config, log_path, compose_project)
        except KeyboardInterrupt:
            return_code = 130
            interrupted = True
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    "\nCampaign interrupted by user; this run will be repeated on restart.\n"
                )
        except Exception as exc:
            return_code = 1
            startup_error = str(exc)
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\nRunner error: {exc}\n")

        duration = time.time() - started
        summary_path = archive_outputs(run_dir, config, work_dir)
        if interrupted:
            status = "interrupted"
        elif return_code == 0 and summary_path:
            status = "ok"
        elif startup_error:
            status = "failed(startup)"
        elif return_code:
            status = f"failed({return_code})"
        else:
            status = "failed(missing-summary)"

        selector, compressor, hdh = spec.states
        metadata = config["campaign_metadata"]
        append_index_row(
            index_path,
            {
                "Run ID": spec.run_id,
                "Execution Mode": mode,
                "Wave": spec.repeat,
                "Repeat": spec.repeat,
                "Dataset": spec.dataset,
                "Model": spec.model,
                "Clients": spec.client_count,
                "Low-spec Percent": spec.low_spec_percentage,
                "Realized Low-spec Percent": f"{metadata['realized_low_spec_percentage']:.6g}",
                "High-spec Percent": spec.high_spec_percentage,
                "Realized High-spec Percent": f"{metadata['realized_high_spec_percentage']:.6g}",
                "High-spec Clients": metadata["high_spec_clients"],
                "Low-spec Clients": metadata["low_spec_clients"],
                "Non-IID Percent": spec.non_iid_percentage,
                "Realized Non-IID Percent": f"{metadata['realized_non_iid_percentage']:.6g}",
                "Non-IID Clients": metadata["non_iid_clients"],
                "IID Clients": metadata["iid_clients"],
                "Dirichlet Alpha": spec.alpha,
                "Delay Percent": spec.delay_percentage,
                "Delayed Clients": metadata["delayed_clients"],
                "Realized Delay Percent": f"{metadata['realized_delay_percentage']:.6g}",
                "Configuration": spec.configuration,
                "Configuration Label": CONFIGURATION_LABELS[spec.configuration],
                "Client Selector": selector,
                "Message Compressor": compressor,
                "HDH": hdh,
                "Partition Seed": config["partition_seed"],
                "Status": status,
                "Duration Seconds": f"{duration:.1f}",
                "Output Dir": str(run_dir),
                "ML Summary CSV": str(summary_path) if summary_path else "",
                "Hostname": hostname,
                "Git Commit": commit,
            },
        )

        if interrupted:
            print(
                f"Campaign interrupted. Completed runs remain archived; "
                f"{spec.run_id} will restart from round 1 next time.",
                file=sys.stderr,
            )
            raise KeyboardInterrupt
        if status == "ok":
            total_rows = regenerate_combined_dataset(phase_indexes, combined_path)
            print(f"OK ({duration:.1f}s). Combined dataset now has {total_rows} rows.")
        else:
            failures += 1
            detail = f" ({startup_error})" if startup_error else ""
            print(f"FAILED: {status}{detail}; see {log_path}", file=sys.stderr)
            if not continue_on_error:
                break

    return failures


def parse_csv_list(raw: str, allowed: Iterable[str], option: str) -> tuple[str, ...]:
    allowed_values = tuple(allowed)
    if not raw:
        return allowed_values
    requested = tuple(item.strip() for item in raw.split(";") if item.strip())
    unknown = [item for item in requested if item not in allowed_values]
    if unknown:
        raise ValueError(f"Unknown value(s) for {option}: {', '.join(unknown)}")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AP4Fed pattern-stress matrix without the GUI."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--through-wave",
        type=int,
        help="Stop the Docker campaign after this repeat wave.",
    )
    parser.add_argument(
        "--only-wave",
        type=int,
        help="Run only one repetition wave; intended for independent Slurm array tasks.",
    )
    parser.add_argument(
        "--configurations",
        default="",
        help="Optional semicolon-separated configuration filter in CS,MC,HDH order.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--docker-project", default="ap4fed-adept")
    parser.add_argument("--host-cpus", type=int, default=HOST_CPU_CAPACITY)
    parser.add_argument("--server-cpus", type=int, default=SERVER_CPUS)
    parser.add_argument("--low-spec-cpus", type=int, default=LOW_SPEC_CPUS)
    parser.add_argument("--high-spec-cpus", type=int, default=HIGH_SPEC_CPUS)
    parser.add_argument(
        "--container-runtime",
        choices=("docker", "singularity"),
        default="docker",
        help="Container orchestrator used for the Docker-defined experiment environment.",
    )
    parser.add_argument(
        "--singularity-image",
        type=Path,
        default=ROOT / "leonardo" / "ap4fed.sif",
    )
    parser.add_argument(
        "--campaign-machine-id",
        help="Stable machine identity used when jobs can resume on different compute nodes.",
    )
    parser.add_argument(
        "--skip-dataset-prefetch",
        action="store_true",
        help="Skip the AG_NEWS and CIFAR-10 cache checks before starting Flower.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def install_interruption_handlers() -> None:
    def stop_safely(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_safely)


def main() -> int:
    install_interruption_handlers()
    args = parse_args()
    if args.rounds < 1:
        print("--rounds must be >= 1", file=sys.stderr)
        return 2

    try:
        configure_resource_profile(
            args.host_cpus,
            args.server_cpus,
            args.low_spec_cpus,
            args.high_spec_cpus,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        configurations = parse_csv_list(args.configurations, CONFIGURATIONS, "--configurations")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    docker_index = phase_index_path(output_dir, "Docker")
    phase_indexes = (("Docker", docker_index),)
    combined_path = output_dir / "adept_experiments.csv"
    try:
        complete_matrix = build_plan(
            set(),
            target_repeats=args.target_repeats,
            configurations=configurations,
            only_wave=args.only_wave,
            execution_mode="Docker",
        )
        if tuple(configurations) == CONFIGURATIONS and args.only_wave is None:
            validate_complete_matrix(complete_matrix, args.target_repeats)
        full_docker_plan = build_plan(
            verified_successful_run_ids(docker_index),
            target_repeats=args.target_repeats,
            configurations=configurations,
            only_wave=args.only_wave,
            execution_mode="Docker",
        )
        docker_plan = build_plan(
            verified_successful_run_ids(docker_index),
            target_repeats=args.target_repeats,
            configurations=configurations,
            through_wave=args.through_wave,
            only_wave=args.only_wave,
            execution_mode="Docker",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        f"Verified completed Docker runs: "
        f"{len(complete_matrix) - len(full_docker_plan)} / {len(complete_matrix)}"
    )
    print_plan("Docker", docker_plan)

    if args.dry_run:
        return 0

    try:
        machine_id = args.campaign_machine_id
        if args.container_runtime == "singularity" and not machine_id:
            machine_id = os.environ.get("SLURM_CLUSTER_NAME", "leonardo")
        hostname, commit = lock_campaign_to_machine(output_dir, machine_id)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.container_runtime == "singularity":
        if not os.environ.get("SLURM_JOB_ID"):
            print("The Singularity runtime must be launched inside a Slurm job.", file=sys.stderr)
            return 2
        if not args.singularity_image.resolve().is_file():
            print(
                f"Singularity image not found: {args.singularity_image.resolve()}",
                file=sys.stderr,
            )
            return 2

    if docker_plan and not args.skip_dataset_prefetch:
        try:
            if args.container_runtime == "singularity":
                ensure_datasets_available_singularity(args.singularity_image)
            else:
                ensure_datasets_available(TASKS)
        except Exception as exc:
            print(f"Unable to prepare datasets: {exc}", file=sys.stderr)
            return 2

    config_paths = (DOCKER_CONFIG_PATH,)
    original_configs = {
        path: path.read_bytes() if path.exists() else None for path in config_paths
    }
    original_compose = (
        DOCKER_ADEPT_COMPOSE_PATH.read_bytes() if DOCKER_ADEPT_COMPOSE_PATH.exists() else None
    )
    docker_failures = 0
    interrupted = False
    try:
        docker_plan = build_plan(
            verified_successful_run_ids(docker_index),
            target_repeats=args.target_repeats,
            configurations=configurations,
            through_wave=args.through_wave,
            only_wave=args.only_wave,
            execution_mode="Docker",
        )
        try:
            docker_failures = execute_phase(
                "Docker",
                docker_plan,
                output_dir,
                phase_indexes,
                combined_path,
                args.rounds,
                hostname,
                commit,
                args.continue_on_error,
                args.docker_project,
                args.container_runtime,
                args.singularity_image,
            )
        except KeyboardInterrupt:
            interrupted = True
    finally:
        for path, original_content in original_configs.items():
            if original_content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original_content)
        if original_compose is None:
            DOCKER_ADEPT_COMPOSE_PATH.unlink(missing_ok=True)
        else:
            DOCKER_ADEPT_COMPOSE_PATH.write_bytes(original_compose)

    total_rows = regenerate_combined_dataset(phase_indexes, combined_path)
    if interrupted:
        remaining_docker = build_plan(
            verified_successful_run_ids(docker_index),
            target_repeats=args.target_repeats,
            configurations=configurations,
            only_wave=args.only_wave,
            execution_mode="Docker",
        )
        print(
            f"Campaign stopped safely: {len(remaining_docker)} Docker run(s) remain. "
            f"Rerun the same command to resume from the first incomplete run."
        )
        return 130
    if docker_failures:
        print(
            f"Completed with {docker_failures} Docker failure(s).",
            file=sys.stderr,
        )
        return 1
    remaining_docker = build_plan(
        verified_successful_run_ids(docker_index),
        target_repeats=args.target_repeats,
        configurations=configurations,
        only_wave=args.only_wave,
        execution_mode="Docker",
    )
    if remaining_docker:
        print(
            f"Campaign paused: {len(remaining_docker)} Docker run(s) remain. "
            f"Combined dataset: {combined_path} ({total_rows} rows)"
        )
        return 0
    print(f"\nDocker campaign complete on {hostname}. Combined dataset: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
