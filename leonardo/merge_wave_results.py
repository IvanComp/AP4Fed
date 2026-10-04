#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

WAVES = 10
RUNS_PER_WAVE = 138
EXPECTED_RUNS = WAVES * RUNS_PER_WAVE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve()

    frames = []
    complete_ids = set()
    wave_counts = {}
    problems = []
    for wave in range(1, WAVES + 1):
        wave_root = root / f"wave_{wave}"
        index_path = wave_root / "docker" / "index.csv"
        combined_path = wave_root / "adept_experiments.csv"
        if not index_path.is_file():
            wave_counts[str(wave)] = 0
            problems.append(f"wave {wave}: missing index")
            continue

        index = pd.read_csv(index_path, dtype=str).fillna("")
        latest = index.drop_duplicates("Run ID", keep="last")
        valid_ids = set()
        for _, record in latest[latest["Status"] == "ok"].iterrows():
            output_dir = Path(record["Output Dir"])
            summary_path = Path(record["ML Summary CSV"])
            if (
                output_dir.is_dir()
                and (output_dir / "config.json").is_file()
                and summary_path.is_file()
            ):
                valid_ids.add(record["Run ID"])
        wave_counts[str(wave)] = len(valid_ids)
        complete_ids.update(valid_ids)
        if len(valid_ids) != RUNS_PER_WAVE:
            problems.append(
                f"wave {wave}: {len(valid_ids)}/{RUNS_PER_WAVE} verified runs"
            )
        if combined_path.is_file():
            frames.append(pd.read_csv(combined_path))

    destination = root / "adept_experiments_all_waves.csv"
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        if "run_id" in combined.columns:
            combined = combined.drop_duplicates("run_id", keep="last")
        combined.to_csv(destination, index=False)
    else:
        pd.DataFrame().to_csv(destination, index=False)

    report = {
        "verified_runs": len(complete_ids),
        "expected_runs": EXPECTED_RUNS,
        "complete": len(complete_ids) == EXPECTED_RUNS and not problems,
        "wave_counts": wave_counts,
        "problems": problems,
        "combined_csv": str(destination),
    }
    report_path = root / "campaign_verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
