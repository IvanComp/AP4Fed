#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from run_adept_campaign import (
    regenerate_client_round_dataset,
    regenerate_combined_dataset,
)


WAVES = 10
RUNS_PER_WAVE = 96
EXPECTED_RUNS = WAVES * RUNS_PER_WAVE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve()

    frames = []
    client_round_frames = []
    complete_ids = set()
    wave_counts = {}
    problems = []
    for wave in range(1, WAVES + 1):
        wave_root = root / f"wave_{wave}"
        index_path = wave_root / "docker" / "index.csv"
        combined_path = wave_root / "adept_experiments.csv"
        client_round_path = wave_root / "adept_experiments_client_rounds.csv"
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
        regenerate_combined_dataset((("Docker", index_path),), combined_path)
        regenerate_client_round_dataset((("Docker", index_path),), client_round_path)
        if combined_path.is_file():
            frames.append(pd.read_csv(combined_path))
        if client_round_path.is_file():
            client_round_frames.append(pd.read_csv(client_round_path))

    destination = root / "adept_experiments_all_waves_combinedAP.csv"
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        if "run_id" in combined.columns:
            combined = combined.drop_duplicates("run_id", keep="last")
        combined.to_csv(destination, index=False)
    else:
        pd.DataFrame().to_csv(destination, index=False)

    client_round_destination = root / "adept_experiments_all_waves_combinedAP_client_rounds.csv"
    if client_round_frames:
        client_round_combined = pd.concat(client_round_frames, ignore_index=True, sort=False)
        client_round_combined = client_round_combined.drop_duplicates(
            ["run_id", "FL Round", "Client ID"], keep="last"
        )
        client_round_combined.to_csv(client_round_destination, index=False)
    else:
        pd.DataFrame().to_csv(client_round_destination, index=False)

    report = {
        "verified_runs": len(complete_ids),
        "expected_runs": EXPECTED_RUNS,
        "complete": len(complete_ids) == EXPECTED_RUNS and not problems,
        "wave_counts": wave_counts,
        "problems": problems,
        "combined_csv": str(destination),
        "client_round_csv": str(client_round_destination),
        "client_round_rows": len(client_round_combined) if client_round_frames else 0,
    }
    report_path = root / "campaign_verification_combinedAP.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
