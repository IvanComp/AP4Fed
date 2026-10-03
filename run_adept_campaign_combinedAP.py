#!/usr/bin/env python3
"""Run the AP4Fed campaign with pairs of architectural patterns enabled."""

from pathlib import Path

import run_adept_campaign as base


COMBINED_CONFIGURATIONS = (
    "ON,ON,OFF",
    "ON,OFF,ON",
    "OFF,ON,ON",
)

COMBINED_CONFIGURATION_LABELS = {
    "ON,ON,OFF": "CS + MC",
    "ON,OFF,ON": "CS + HDH",
    "OFF,ON,ON": "MC + HDH",
}


def combined_stress_profiles(
    configuration: str,
) -> tuple[tuple[int, int, float, int], ...]:
    """Return synchronized 25/50/75 stress for each enabled pattern pair."""
    profiles = []
    nominal = base.NOMINAL_STRESS_PERCENTAGE
    nominal_high = 100 - nominal
    for stress in (25, 50, 75):
        if configuration == "ON,ON,OFF":  # Client Selector + Message Compressor
            profiles.append((100 - stress, nominal, base.REFERENCE_ALPHA, stress))
        elif configuration == "ON,OFF,ON":  # Client Selector + HDH
            profiles.append((100 - stress, stress, base.REFERENCE_ALPHA, nominal))
        elif configuration == "OFF,ON,ON":  # Message Compressor + HDH
            profiles.append((nominal_high, stress, base.REFERENCE_ALPHA, stress))
        else:
            raise ValueError(f"Unsupported combined pattern configuration: {configuration}")
    return tuple(profiles)


def configure_combined_campaign() -> None:
    base.CONFIGURATIONS = COMBINED_CONFIGURATIONS
    base.CONFIGURATION_LABELS = COMBINED_CONFIGURATION_LABELS
    base.DEFAULT_OUTPUT_DIR = Path("pattern_stress_combinedAP_docker_results")
    base.stress_profiles = combined_stress_profiles


def main() -> int:
    configure_combined_campaign()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
