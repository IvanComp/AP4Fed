#!/usr/bin/env python3
"""Run the standard and combined-pattern AP4Fed campaigns with 6 clients.

The matrices are identical to run_adept_campaign.py and
run_adept_campaign_combinedAP.py (patterns, stress profiles, datasets, seeds,
rounds, local epochs); only the federation size changes, completing the
4, 6, 8, 10 client scale. Results keep the same schema and run identifiers
(``__n6__``), so they merge with the existing campaign CSV files.
"""

import sys

import run_adept_campaign as base
import run_adept_campaign_combinedAP as combined


SIX_CLIENT_COUNTS = (6,)
CAMPAIGNS = ("single", "combinedAP")
_build_plan = base.build_plan


def build_six_client_plan(*args, **kwargs):
    kwargs["client_counts"] = SIX_CLIENT_COUNTS
    return _build_plan(*args, **kwargs)


def configure_six_client_campaign(campaign: str) -> None:
    if campaign not in CAMPAIGNS:
        raise ValueError(f"--campaign must be one of {', '.join(CAMPAIGNS)}")
    if campaign == "combinedAP":
        combined.configure_combined_campaign()
        base.DEFAULT_OUTPUT_DIR = base.Path("pattern_stress_combinedAP_6clients_docker_results")
    else:
        base.DEFAULT_OUTPUT_DIR = base.Path("pattern_stress_6clients_docker_results")
    base.CLIENT_COUNTS = SIX_CLIENT_COUNTS
    base.build_plan = build_six_client_plan


def pop_campaign_argument(argv: list[str]) -> str:
    campaign = "single"
    for index, value in enumerate(argv):
        if value == "--campaign" and index + 1 < len(argv):
            campaign = argv[index + 1]
            del argv[index : index + 2]
            break
        if value.startswith("--campaign="):
            campaign = value.split("=", 1)[1]
            del argv[index]
            break
    return campaign


def main() -> int:
    try:
        configure_six_client_campaign(pop_campaign_argument(sys.argv))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
