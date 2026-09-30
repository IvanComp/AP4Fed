#!/usr/bin/env python3
"""Run the static AP4Fed pattern matrix used by the ADEPT FL pilot.

Runs are scheduled breadth-first by repeat: all missing cells receive repeat 1
before any cell receives repeat 2, and so on.  The simulations themselves stay
serial so concurrent workloads cannot contaminate performance measurements.
The Local phase must finish before the Docker phase starts on the same Linux
host.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import zlib
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
DEFAULT_EXISTING_RESULTS = ROOT / "tests" / "data" / "FLwithAP_MLdata_split.csv"
DEFAULT_OUTPUT_DIR = ROOT / "adept_campaign_results"

MODELS = ("CNN 16k", "squeezenet1_1")
CONFIGURATIONS = (
    "OFF,OFF,OFF",
    "ON,OFF,OFF",
    "OFF,ON,OFF",
    "OFF,OFF,ON",
    "ON,ON,OFF",
    "ON,OFF,ON",
    "OFF,ON,ON",
    "ON,ON,ON",
)

INDEX_FIELDS = (
    "Run ID",
    "Execution Mode",
    "Wave",
    "Repeat",
    "Model",
    "Configuration",
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

CLIENT_TEMPLATE = (
    (1, 3, "IID", 1.0),
    (2, 3, "IID", 1.0),
    (3, 3, "IID", 1.0),
    (4, 3, "non-IID", 0.5),
    (5, 1, "non-IID", 0.5),
)


@dataclass(frozen=True)
class RunSpec:
    repeat: int
    model: str
    configuration: str
    execution_mode: str = "Local"

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
    def run_id(self) -> str:
        model_slug = sanitize_name(self.model.lower())
        config_slug = self.configuration.lower().replace(",", "_")
        return f"adept__{self.mode.lower()}__{model_slug}__{config_slug}__r{self.repeat:02d}"


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def read_results_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame.columns) == 1:
        frame = pd.read_csv(path, sep=";", decimal=",")
    return frame


def split_pattern_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "config_id" in result.columns:
        return result

    ap_column = next((column for column in result.columns if str(column).startswith("AP List")), None)
    if ap_column is None:
        raise ValueError("Results CSV has neither 'config_id' nor an 'AP List' column")

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
    result["config_id"] = (
        result["client_selector_pattern"]
        + ","
        + result["message_compressor_pattern"]
        + ","
        + result["hdh_pattern"]
    )
    return result.drop(columns=[ap_column])


def existing_counts(path: Path, models: Iterable[str], configurations: Iterable[str]) -> dict[tuple[str, str], int]:
    frame = split_pattern_column(read_results_csv(path))
    if "Model" not in frame.columns:
        raise ValueError("Existing results CSV has no 'Model' column")

    allowed_models = set(models)
    allowed_configurations = set(configurations)
    counts: dict[tuple[str, str], int] = {}
    for (configuration, model), count in frame.groupby(["config_id", "Model"]).size().items():
        if configuration in allowed_configurations and model in allowed_models:
            counts[(str(configuration), str(model))] = int(count)
    return counts


def successful_run_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    frame = pd.read_csv(index_path, dtype=str).fillna("")
    if "Run ID" not in frame or "Status" not in frame:
        return set()
    return set(frame.loc[frame["Status"] == "ok", "Run ID"].astype(str))


def build_plan(
    counts: dict[tuple[str, str], int],
    completed_run_ids: set[str],
    target_repeats: int,
    models: Iterable[str] = MODELS,
    configurations: Iterable[str] = CONFIGURATIONS,
    through_wave: int | None = None,
    execution_mode: str = "Local",
) -> list[RunSpec]:
    if target_repeats < 1:
        raise ValueError("target_repeats must be >= 1")
    if through_wave is not None and through_wave < 1:
        raise ValueError("through_wave must be >= 1")

    last_wave = min(target_repeats, through_wave) if through_wave else target_repeats
    plan: list[RunSpec] = []
    for repeat in range(1, last_wave + 1):
        for configuration in configurations:
            for model in models:
                spec = RunSpec(
                    repeat=repeat,
                    model=model,
                    configuration=configuration,
                    execution_mode=execution_mode,
                )
                if repeat <= counts.get((configuration, model), 0):
                    continue
                if spec.run_id in completed_run_ids:
                    continue
                plan.append(spec)
    return plan


def build_partition_seed(spec: RunSpec, rounds: int) -> int:
    payload = json.dumps(
        {
            "model": spec.model,
            "configuration": spec.configuration,
            "repeat": spec.repeat,
            "rounds": rounds,
        },
        sort_keys=True,
    ).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def build_patterns(spec: RunSpec) -> dict[str, dict[str, Any]]:
    selector, compressor, hdh = spec.states
    return {
        "client_registry": {"enabled": True, "params": {}},
        "client_selector": {
            "enabled": selector == "ON",
            "params": {
                "selection_strategy": "Resource-Based",
                "selection_criteria": "CPU",
                "selection_value": 2,
            },
        },
        "client_cluster": {"enabled": False, "params": {}},
        "message_compressor": {"enabled": compressor == "ON", "params": {}},
        "model_co-versioning_registry": {"enabled": False, "params": {}},
        "multi-task_model_trainer": {"enabled": False, "params": {}},
        "heterogeneous_data_handler": {"enabled": hdh == "ON", "params": {}},
    }


def build_config(spec: RunSpec, rounds: int) -> dict[str, Any]:
    clients = [
        {
            "client_id": client_id,
            "cpu": cpu,
            "ram": 2,
            "dataset": "CIFAR-10",
            "data_distribution_type": distribution,
            "non_iid_alpha": alpha,
            "data_persistence_type": "Same Data",
            "delay_combobox": "No",
            "delay_min_seconds": 0,
            "delay_max_seconds": 0,
            "model": spec.model,
            "epochs": 1,
        }
        for client_id, cpu, distribution, alpha in CLIENT_TEMPLATE
    ]
    return {
        "simulation_type": spec.mode,
        "rounds": int(rounds),
        "clients": len(clients),
        "clients_per_round": len(clients),
        "dataset": "CIFAR-10",
        "adaptation": "None",
        "LLM": "llama3.2:3b",
        "partition_seed": build_partition_seed(spec, rounds),
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


def ensure_cifar10_available() -> None:
    """Download CIFAR-10 once, then expose the same cache to Docker."""
    from torchvision.datasets import CIFAR10

    data_root = LOCAL_DIR / "data"
    print(f"Checking CIFAR-10 dataset cache in {data_root} ...")
    CIFAR10(root=str(data_root), train=True, download=True)
    CIFAR10(root=str(data_root), train=False, download=True)
    docker_data_root = DOCKER_DIR / "data"
    shutil.copytree(data_root, docker_data_root, dirs_exist_ok=True)
    print("CIFAR-10 dataset cache is ready for Local and Docker.")


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


def _preliminary_with_metadata(path: Path) -> pd.DataFrame:
    frame = split_pattern_column(read_results_csv(path))
    frame = frame.copy()
    frame["repeat"] = frame.groupby(["config_id", "Model"]).cumcount() + 1
    frame["run_id"] = [
        f"preliminary__local__{sanitize_name(str(model).lower())}__{str(configuration).lower().replace(',', '_')}__r{repeat:02d}"
        for model, configuration, repeat in zip(frame["Model"], frame["config_id"], frame["repeat"])
    ]
    frame["result_source"] = "preliminary"
    frame["execution_mode"] = "Local"
    return frame


def regenerate_combined_dataset(
    existing_path: Path,
    phase_indexes: Iterable[tuple[str, Path]],
    destination: Path,
) -> int:
    frames = [_preliminary_with_metadata(existing_path)]
    for mode, index_path in phase_indexes:
        if not index_path.exists():
            continue
        index = pd.read_csv(index_path, dtype=str).fillna("")
        successful = index[index["Status"] == "ok"].drop_duplicates("Run ID", keep="last")
        for _, record in successful.iterrows():
            summary_path = Path(record["ML Summary CSV"])
            if not summary_path.exists():
                continue
            frame = split_pattern_column(read_results_csv(summary_path))
            frame["repeat"] = int(record["Repeat"])
            frame["run_id"] = record["Run ID"]
            frame["result_source"] = "campaign"
            frame["execution_mode"] = mode
            frame["hostname"] = record.get("Hostname", "")
            frame["git_commit"] = record.get("Git Commit", "")
            frame["partition_seed"] = int(record["Partition Seed"])
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
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

    server.setdefault("environment", {})["NUM_ROUNDS"] = str(config["rounds"])
    generated_services = {"server": server}
    for detail in config["client_details"]:
        client_id = int(detail["client_id"])
        cpu = int(detail["cpu"])
        ram = int(detail["ram"])
        service = copy.deepcopy(client_template)
        service.pop("deploy", None)
        service["container_name"] = f"Client{client_id}"
        service["cpus"] = cpu
        service["mem_limit"] = f"{ram}g"
        environment = service.setdefault("environment", {})
        environment.update(
            {
                "CLIENT_ID": str(client_id),
                "NUM_CPUS": str(cpu),
                "NUM_RAM": str(ram),
                "NUM_ROUNDS": str(config["rounds"]),
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


def lock_campaign_to_machine(output_dir: Path) -> tuple[str, str]:
    if platform.system() != "Linux":
        raise RuntimeError("The ADEPT campaign is locked to the Linux workstation and cannot run on this computer.")

    hostname = socket.gethostname()
    commit = git_commit()
    manifest_path = output_dir / "campaign_machine.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_hostname = str(manifest.get("hostname", ""))
        if recorded_hostname and recorded_hostname != hostname:
            raise RuntimeError(
                f"Campaign belongs to host '{recorded_hostname}', not current host '{hostname}'."
            )
    else:
        write_json(
            manifest_path,
            {
                "hostname": hostname,
                "platform": platform.platform(),
                "git_commit_at_start": commit,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "phase_order": ["Local", "Docker"],
            },
        )
    return hostname, commit


def phase_index_path(output_dir: Path, mode: str) -> Path:
    return output_dir / mode.lower() / "index.csv"


def print_plan(mode: str, plan: list[RunSpec]) -> None:
    print(f"\n{mode} phase: {len(plan)} scheduled run(s)")
    for wave in sorted({spec.repeat for spec in plan}):
        wave_specs = [spec for spec in plan if spec.repeat == wave]
        print(f"Wave {wave}: {len(wave_specs)} run(s)")
        for spec in wave_specs:
            print(f"- {spec.run_id}: {spec.model} / {spec.configuration}")


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
            process = subprocess.run(
                base_command
                + [
                    "up",
                    "--build",
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


def execute_phase(
    mode: str,
    plan: list[RunSpec],
    output_dir: Path,
    existing_results: Path,
    phase_indexes: tuple[tuple[str, Path], ...],
    combined_path: Path,
    rounds: int,
    hostname: str,
    commit: str,
    continue_on_error: bool,
    compose_project: str,
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
        log_path = run_dir / ("flower.log" if mode == "Local" else "docker-compose.log")
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[{mode} {position}/{len(plan)}] Wave {spec.repeat}: "
            f"{spec.model} / {spec.configuration}"
        )

        reset_runtime_state(work_dir, mode)
        write_json(config_path, config)
        started = time.time()
        startup_error = ""
        try:
            if mode == "Local":
                return_code = run_local_process(config, log_path, rounds)
            else:
                return_code = run_docker_process(config, log_path, compose_project)
        except Exception as exc:
            return_code = 1
            startup_error = str(exc)
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\nRunner error: {exc}\n")

        duration = time.time() - started
        summary_path = archive_outputs(run_dir, config, work_dir)
        if return_code == 0 and summary_path:
            status = "ok"
        elif startup_error:
            status = "failed(startup)"
        elif return_code:
            status = f"failed({return_code})"
        else:
            status = "failed(missing-summary)"

        selector, compressor, hdh = spec.states
        append_index_row(
            index_path,
            {
                "Run ID": spec.run_id,
                "Execution Mode": mode,
                "Wave": spec.repeat,
                "Repeat": spec.repeat,
                "Model": spec.model,
                "Configuration": spec.configuration,
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

        if status == "ok":
            total_rows = regenerate_combined_dataset(existing_results, phase_indexes, combined_path)
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
        description="Complete the ADEPT FL matrix breadth-first by repeat, without the GUI."
    )
    parser.add_argument("--existing-results", type=Path, default=DEFAULT_EXISTING_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--through-wave",
        type=int,
        help=(
            "Stop the Local phase after this repeat wave. Docker starts only when the full "
            "Local target is complete."
        ),
    )
    parser.add_argument(
        "--models",
        default="",
        help="Optional semicolon-separated model filter (commas are reserved inside configuration IDs).",
    )
    parser.add_argument(
        "--configurations",
        default="",
        help="Optional semicolon-separated configuration filter, e.g. 'ON,ON,OFF;ON,ON,ON'.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--docker-project", default="ap4fed-adept")
    parser.add_argument(
        "--skip-dataset-prefetch",
        action="store_true",
        help="Skip the single-process CIFAR-10 cache check before starting Flower.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        print("--rounds must be >= 1", file=sys.stderr)
        return 2

    try:
        models = parse_csv_list(args.models, MODELS, "--models")
        configurations = parse_csv_list(args.configurations, CONFIGURATIONS, "--configurations")
        counts = existing_counts(args.existing_results.resolve(), models, configurations)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    existing_results = args.existing_results.resolve()
    output_dir = args.output_dir.resolve()
    local_index = phase_index_path(output_dir, "Local")
    docker_index = phase_index_path(output_dir, "Docker")
    phase_indexes = (("Local", local_index), ("Docker", docker_index))
    combined_path = output_dir / "adept_experiments.csv"
    try:
        full_local_plan = build_plan(
            counts,
            successful_run_ids(local_index),
            target_repeats=args.target_repeats,
            models=models,
            configurations=configurations,
            execution_mode="Local",
        )
        local_plan = build_plan(
            counts,
            successful_run_ids(local_index),
            target_repeats=args.target_repeats,
            models=models,
            configurations=configurations,
            through_wave=args.through_wave,
            execution_mode="Local",
        )
        docker_plan = build_plan(
            {},
            successful_run_ids(docker_index),
            target_repeats=args.target_repeats,
            models=models,
            configurations=configurations,
            execution_mode="Docker",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Existing preliminary experiments: {sum(counts.values())}")
    print_plan("Local", local_plan)
    if {spec.run_id for spec in local_plan} == {spec.run_id for spec in full_local_plan}:
        print_plan("Docker (after Local)", docker_plan)
    else:
        print(f"\nDocker phase pending until the remaining {len(full_local_plan)} Local run(s) are complete.")

    if args.dry_run:
        return 0

    try:
        hostname, commit = lock_campaign_to_machine(output_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if (local_plan or docker_plan) and not args.skip_dataset_prefetch:
        try:
            ensure_cifar10_available()
        except Exception as exc:
            print(f"Unable to prepare CIFAR-10: {exc}", file=sys.stderr)
            return 2

    config_paths = (LOCAL_CONFIG_PATH, DOCKER_CONFIG_PATH)
    original_configs = {
        path: path.read_bytes() if path.exists() else None for path in config_paths
    }
    runtime_state_paths = (LOCAL_DIR / ".client_idx", LOCAL_DIR / ".cpu_pool_state.json")
    original_runtime_state = {
        path: path.read_bytes() if path.exists() else None for path in runtime_state_paths
    }
    original_compose = (
        DOCKER_ADEPT_COMPOSE_PATH.read_bytes() if DOCKER_ADEPT_COMPOSE_PATH.exists() else None
    )
    local_failures = 0
    docker_failures = 0
    try:
        local_failures = execute_phase(
            "Local",
            local_plan,
            output_dir,
            existing_results,
            phase_indexes,
            combined_path,
            args.rounds,
            hostname,
            commit,
            args.continue_on_error,
            args.docker_project,
        )

        remaining_local = build_plan(
            counts,
            successful_run_ids(local_index),
            target_repeats=args.target_repeats,
            models=models,
            configurations=configurations,
            execution_mode="Local",
        )
        if local_failures or remaining_local:
            print(
                f"Docker not started: Local phase still has {len(remaining_local)} pending run(s) "
                f"and {local_failures} failure(s)."
            )
        else:
            docker_plan = build_plan(
                {},
                successful_run_ids(docker_index),
                target_repeats=args.target_repeats,
                models=models,
                configurations=configurations,
                execution_mode="Docker",
            )
            docker_failures = execute_phase(
                "Docker",
                docker_plan,
                output_dir,
                existing_results,
                phase_indexes,
                combined_path,
                args.rounds,
                hostname,
                commit,
                args.continue_on_error,
                args.docker_project,
            )
    finally:
        for path, original_content in original_configs.items():
            if original_content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original_content)
        for path, original_content in original_runtime_state.items():
            if original_content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original_content)
        if original_compose is None:
            DOCKER_ADEPT_COMPOSE_PATH.unlink(missing_ok=True)
        else:
            DOCKER_ADEPT_COMPOSE_PATH.write_bytes(original_compose)

    total_rows = regenerate_combined_dataset(existing_results, phase_indexes, combined_path)
    if local_failures or docker_failures:
        print(
            f"Completed with {local_failures} Local and {docker_failures} Docker failure(s).",
            file=sys.stderr,
        )
        return 1
    remaining_local = build_plan(
        counts,
        successful_run_ids(local_index),
        target_repeats=args.target_repeats,
        models=models,
        configurations=configurations,
        execution_mode="Local",
    )
    remaining_docker = build_plan(
        {},
        successful_run_ids(docker_index),
        target_repeats=args.target_repeats,
        models=models,
        configurations=configurations,
        execution_mode="Docker",
    )
    if remaining_local or remaining_docker:
        print(
            f"Campaign paused: {len(remaining_local)} Local and {len(remaining_docker)} Docker run(s) remain. "
            f"Combined dataset: {combined_path} ({total_rows} rows)"
        )
        return 0
    print(f"\nLocal and Docker campaigns complete on {hostname}. Combined dataset: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
