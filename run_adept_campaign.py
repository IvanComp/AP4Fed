#!/usr/bin/env python3
"""Run the static AP4Fed pattern matrix used by the ADEPT FL pilot.

Runs are scheduled breadth-first by repeat: all missing cells receive repeat 1
before any cell receives repeat 2, and so on.  The simulations themselves stay
serial so concurrent workloads cannot contaminate performance measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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
LOCAL_CONFIG_PATH = LOCAL_DIR / "configuration" / "config.json"
DEFAULT_EXISTING_RESULTS = (
    ROOT.parent / "patterns-sa" / "federatedlearning" / "FLwithAP_MLdata_split.csv"
)
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
        return f"adept__{model_slug}__{config_slug}__r{self.repeat:02d}"


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
                spec = RunSpec(repeat=repeat, model=model, configuration=configuration)
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
        "simulation_type": "Local",
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


def reset_local_state() -> None:
    for folder_name in ("performance", "performance_MLdata", "logs"):
        folder = LOCAL_DIR / folder_name
        if not folder.exists():
            continue
        for entry in folder.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    (LOCAL_DIR / ".client_idx").write_text("0", encoding="utf-8")
    (LOCAL_DIR / ".cpu_pool_state.json").write_text('{"allocations": {}}\n', encoding="utf-8")


def ensure_cifar10_available() -> None:
    """Download/verify CIFAR-10 once before Ray starts multiple clients."""
    from torchvision.datasets import CIFAR10

    data_root = LOCAL_DIR / "data"
    print(f"Checking CIFAR-10 dataset cache in {data_root} ...")
    CIFAR10(root=str(data_root), train=True, download=True)
    CIFAR10(root=str(data_root), train=False, download=True)
    print("CIFAR-10 dataset cache is ready.")


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def archive_outputs(run_dir: Path, config: dict[str, Any]) -> Path | None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    copy_if_exists(LOCAL_DIR / "performance", run_dir / "performance")
    copy_if_exists(LOCAL_DIR / "performance_MLdata", run_dir / "performance_MLdata")
    copy_if_exists(LOCAL_DIR / "logs", run_dir / "logs")
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
        f"preliminary__{sanitize_name(str(model).lower())}__{str(configuration).lower().replace(',', '_')}__r{repeat:02d}"
        for model, configuration, repeat in zip(frame["Model"], frame["config_id"], frame["repeat"])
    ]
    frame["result_source"] = "preliminary"
    return frame


def regenerate_combined_dataset(existing_path: Path, index_path: Path, destination: Path) -> int:
    frames = [_preliminary_with_metadata(existing_path)]
    if index_path.exists():
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
            frame["partition_seed"] = int(record["Partition Seed"])
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return len(combined)


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
        help="Stop after this repeat wave. Use 1 to populate every currently empty cell once.",
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

    output_dir = args.output_dir.resolve()
    index_path = output_dir / "index.csv"
    combined_path = output_dir / "adept_experiments.csv"
    completed = successful_run_ids(index_path)
    try:
        plan = build_plan(
            counts,
            completed,
            target_repeats=args.target_repeats,
            models=models,
            configurations=configurations,
            through_wave=args.through_wave,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Existing preliminary experiments: {sum(counts.values())}")
    print(f"Scheduled new experiments: {len(plan)}")
    for wave in sorted({spec.repeat for spec in plan}):
        wave_specs = [spec for spec in plan if spec.repeat == wave]
        print(f"\nWave {wave}: {len(wave_specs)} run(s)")
        for spec in wave_specs:
            print(f"- {spec.run_id}: {spec.model} / {spec.configuration}")

    if args.dry_run:
        return 0

    if not plan:
        total_rows = regenerate_combined_dataset(args.existing_results.resolve(), index_path, combined_path)
        print(f"Nothing to run. Combined dataset: {combined_path} ({total_rows} rows)")
        return 0

    if not args.skip_dataset_prefetch:
        try:
            ensure_cifar10_available()
        except Exception as exc:
            print(f"Unable to prepare CIFAR-10: {exc}", file=sys.stderr)
            return 2

    original_config = LOCAL_CONFIG_PATH.read_text(encoding="utf-8") if LOCAL_CONFIG_PATH.exists() else None
    runtime_state_paths = (LOCAL_DIR / ".client_idx", LOCAL_DIR / ".cpu_pool_state.json")
    original_runtime_state = {
        path: path.read_bytes() if path.exists() else None for path in runtime_state_paths
    }
    failures = 0
    try:
        for position, spec in enumerate(plan, start=1):
            config = build_config(spec, args.rounds)
            run_dir = output_dir / "runs" / spec.run_id
            log_path = run_dir / "flower.log"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n[{position}/{len(plan)}] Wave {spec.repeat}: "
                f"{spec.model} / {spec.configuration}"
            )
            reset_local_state()
            write_json(LOCAL_CONFIG_PATH, config)

            env = dict(os.environ)
            env["AP4FED_ROUNDS_OVERRIDE"] = str(args.rounds)
            env["PYTHONUNBUFFERED"] = "1"
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.run(
                    ["flower-simulation", "--app", ".", "--num-supernodes", str(config["clients"])],
                    cwd=LOCAL_DIR,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            duration = time.time() - started
            summary_path = archive_outputs(run_dir, config)
            status = "ok" if process.returncode == 0 and summary_path else (
                f"failed({process.returncode})" if process.returncode else "failed(missing-summary)"
            )
            selector, compressor, hdh = spec.states
            append_index_row(
                index_path,
                {
                    "Run ID": spec.run_id,
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
                },
            )

            if status == "ok":
                total_rows = regenerate_combined_dataset(
                    args.existing_results.resolve(), index_path, combined_path
                )
                print(f"OK ({duration:.1f}s). Combined dataset now has {total_rows} rows.")
            else:
                failures += 1
                print(f"FAILED: {status}; see {log_path}", file=sys.stderr)
                if not args.continue_on_error:
                    return process.returncode or 1
    finally:
        if original_config is None:
            LOCAL_CONFIG_PATH.unlink(missing_ok=True)
        else:
            LOCAL_CONFIG_PATH.write_text(original_config, encoding="utf-8")
        for path, original_content in original_runtime_state.items():
            if original_content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original_content)

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1
    print(f"\nCampaign complete. Combined dataset: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
