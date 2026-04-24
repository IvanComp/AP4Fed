#!/usr/bin/env python3
"""500-client variant of the paper experiment builder.

This runner reuses the single-script local campaign/export pipeline, but fixes:
- total clients = 500
- clients per round (k) = 5
- default output dir = Experiments_100r_k5
- default staging dir = paper_results_local_100r_k5
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import build_paper_experiments as base


TOTAL_CLIENTS = 500
CLIENTS_PER_ROUND = 5
LOCAL_OUTPUT_DIR = base.ROOT / "Experiments_100r_k5"
LOCAL_STAGING_DIR = base.ROOT / "paper_results_local_100r_k5"
DOCKER_OUTPUT_DIR = base.ROOT / "Experiments_100r_k5_docker"
DOCKER_STAGING_DIR = base.ROOT / "paper_results_docker_100r_k5"


def build_client_details(total_clients: int = TOTAL_CLIENTS) -> List[Dict[str, Any]]:
    """Scale the existing 5-client heterogeneous setup up to `total_clients`."""
    template = base.CLIENT_TEMPLATE
    details: List[Dict[str, Any]] = []
    for idx in range(total_clients):
        client = copy.deepcopy(template[idx % len(template)])
        client["client_id"] = idx + 1
        details.append(client)
    return details


def build_local_config(
    spec: base.ApproachSpec,
    rounds: int,
    repeat_idx: int,
    ollama_base_url: str,
    simulation_type: str = "Local",
) -> Dict[str, Any]:
    client_details = build_client_details()
    return {
        "simulation_type": simulation_type,
        "rounds": int(rounds),
        "clients": TOTAL_CLIENTS,
        "clients_per_round": CLIENTS_PER_ROUND,
        "dataset": "FashionMNIST",
        "adaptation": spec.adaptation,
        "LLM": spec.llm,
        "ollama_base_url": ollama_base_url,
        "partition_seed": base.build_partition_seed(spec, repeat_idx, rounds),
        "patterns": base.build_patterns(),
        "client_generation_mode": "manual",
        "client_profiles": [],
        "client_details": client_details,
    }


def main() -> int:
    base.LOCAL_OUTPUT_DIR = Path(LOCAL_OUTPUT_DIR)
    base.LOCAL_STAGING_DIR = Path(LOCAL_STAGING_DIR)
    base.DOCKER_OUTPUT_DIR = Path(DOCKER_OUTPUT_DIR)
    base.DOCKER_STAGING_DIR = Path(DOCKER_STAGING_DIR)
    base.build_local_config = build_local_config
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
