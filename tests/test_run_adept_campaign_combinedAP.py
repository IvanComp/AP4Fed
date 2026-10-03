import subprocess
import sys
import unittest

from run_adept_campaign_combinedAP import (
    COMBINED_CONFIGURATIONS,
    combined_stress_profiles,
)


class CombinedCampaignTests(unittest.TestCase):
    def test_each_pair_has_three_synchronized_stress_levels(self):
        self.assertEqual(
            ((75, 25, 0.5, 25), (50, 25, 0.5, 50), (25, 25, 0.5, 75)),
            combined_stress_profiles("ON,ON,OFF"),
        )
        self.assertEqual(
            ((75, 25, 0.5, 25), (50, 50, 0.5, 25), (25, 75, 0.5, 25)),
            combined_stress_profiles("ON,OFF,ON"),
        )
        self.assertEqual(
            ((75, 25, 0.5, 25), (75, 50, 0.5, 50), (75, 75, 0.5, 75)),
            combined_stress_profiles("OFF,ON,ON"),
        )

    def test_dry_run_contains_54_runs_per_wave(self):
        process = subprocess.run(
            [
                sys.executable,
                "run_adept_campaign_combinedAP.py",
                "--dry-run",
                "--target-repeats",
                "1",
                "--skip-dataset-prefetch",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Docker phase: 54 scheduled run(s)", process.stdout)
        self.assertEqual(3, len(COMBINED_CONFIGURATIONS))


if __name__ == "__main__":
    unittest.main()
