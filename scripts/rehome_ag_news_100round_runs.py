#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "paper_campaigns" / "ag_news__mlp"
ARCHIVE_ROOT = ROOT / "paper_campaigns" / "ag_news__mlp_100round"
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_parent(dst)
    shutil.copy2(src, dst)


def append_index_row(index_path: Path, row: dict) -> None:
    ensure_parent(index_path)
    write_header = not index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})


def main() -> int:
    source_staging = SOURCE_ROOT / "staging"
    archive_staging = ARCHIVE_ROOT / "staging"
    archive_experiments = ARCHIVE_ROOT / "experiments"
    archive_staging.mkdir(parents=True, exist_ok=True)
    archive_experiments.mkdir(parents=True, exist_ok=True)

    index_path = source_staging / "index.csv"
    rows = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    kept_rows = []
    moved_rows = []

    for row in rows:
        run_dir = Path(str(row.get("Output Dir", "")).strip())
        config_path = run_dir / "config.initial.json"
        rounds = None
        if config_path.exists():
            try:
                rounds = int(read_json(config_path).get("rounds"))
            except Exception:
                rounds = None

        if rounds == 100:
            moved_rows.append(row)
        else:
            kept_rows.append(row)

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})

    for row in moved_rows:
        run_dir = Path(str(row["Output Dir"]))
        archive_run_dir = archive_staging / run_dir.name
        if run_dir.exists():
            if archive_run_dir.exists():
                shutil.rmtree(archive_run_dir)
            shutil.move(str(run_dir), str(archive_run_dir))

        archive_row = dict(row)
        archive_row["Output Dir"] = str(archive_run_dir)
        ml_src = archive_run_dir / "performance_MLdata" / "FLwithAP_MLdata.csv"
        rationale_src = archive_run_dir / "performance" / "FLwithAP_adaptation_rationales.csv"
        archive_row["ML Summary CSV"] = str(ml_src) if ml_src.exists() else ""
        archive_row["Rationale CSV"] = str(rationale_src) if rationale_src.exists() else ""
        append_index_row(archive_staging / "index.csv", archive_row)

        config = {}
        config_src = archive_run_dir / "config.initial.json"
        if config_src.exists():
            try:
                config = read_json(config_src)
            except Exception:
                config = {}
        cfg_name = str(row.get("Configuration", "")).replace("_", "-")
        repeat = int(row.get("Repeat", "0") or 0)
        exp_dir = archive_experiments / cfg_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        safe_copy(archive_run_dir / "performance" / "FLwithAP_performance_metrics.csv", exp_dir / f"r{repeat}.csv")
        safe_copy(rationale_src, exp_dir / f"r{repeat}_rationales.csv")
        if config_src.exists():
            safe_copy(config_src, exp_dir / "config.json")

        source_exp_dir = SOURCE_ROOT / "experiments" / cfg_name
        for suffix in (f"r{repeat}.csv", f"r{repeat}_rationales.csv"):
            target = source_exp_dir / suffix
            if target.exists():
                target.unlink()

    campaign_info = {
        "task_slug": "ag_news__mlp_100round",
        "dataset": "AG_NEWS",
        "model": "MLP",
        "mode": "docker",
        "kind": "archive",
    }
    for target in (archive_staging / "campaign_info.json", archive_experiments / "campaign_info.json"):
        target.write_text(json.dumps(campaign_info, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
