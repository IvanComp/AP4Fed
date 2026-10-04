import subprocess
import sys
import unittest

from run_adept_campaign import RunSpec
from run_adept_campaign_combinedAP import (
    COMBINED_CONFIGURATIONS,
    combined_stress_profiles,
)


class CombinedCampaignTests(unittest.TestCase):
    def test_combined_profiles_preserve_legacy_and_add_mc_endpoints(self):
        self.assertEqual(
            (
                (75, 25, 0.5, 25),
                (50, 25, 0.5, 50),
                (25, 25, 0.5, 75),
                (75, 25, 0.5, 0),
                (75, 25, 0.5, 100),
            ),
            combined_stress_profiles("ON,ON,OFF"),
        )
        self.assertEqual(
            ((75, 25, 0.5, 25), (50, 50, 0.5, 25), (25, 75, 0.5, 25)),
            combined_stress_profiles("ON,OFF,ON"),
        )
        self.assertEqual(
            (
                (75, 25, 0.5, 25),
                (75, 50, 0.5, 50),
                (75, 75, 0.5, 75),
                (75, 25, 0.5, 0),
                (75, 25, 0.5, 100),
            ),
            combined_stress_profiles("OFF,ON,ON"),
        )
        self.assertEqual(
            (
                (75, 25, 0.5, 25),
                (75, 25, 0.5, 0),
                (75, 25, 0.5, 100),
            ),
            combined_stress_profiles("ON,ON,ON"),
        )

    def test_dry_run_contains_96_runs_per_wave(self):
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
        self.assertIn("Docker phase: 96 scheduled run(s)", process.stdout)
        self.assertEqual(4, len(COMBINED_CONFIGURATIONS))

    def test_legacy_54_run_ids_are_preserved_and_leave_42_new_runs(self):
        legacy_states = ("ON,ON,OFF", "ON,OFF,ON", "OFF,ON,ON")
        legacy_ids = {
            RunSpec(1, dataset, model, clients, high, non_iid, alpha, delay, state).run_id
            for dataset, model in (("AG_NEWS", "MLP"), ("CIFAR-10", "CNN 16k"))
            for clients in (4, 8, 10)
            for state in legacy_states
            for high, non_iid, alpha, delay in combined_stress_profiles(state)[:3]
        }
        expanded_ids = {
            RunSpec(1, dataset, model, clients, high, non_iid, alpha, delay, state).run_id
            for dataset, model in (("AG_NEWS", "MLP"), ("CIFAR-10", "CNN 16k"))
            for clients in (4, 8, 10)
            for state in COMBINED_CONFIGURATIONS
            for high, non_iid, alpha, delay in combined_stress_profiles(state)
        }
        self.assertEqual(54, len(legacy_ids))
        self.assertEqual(96, len(expanded_ids))
        self.assertTrue(legacy_ids.issubset(expanded_ids))
        self.assertEqual(42, len(expanded_ids - legacy_ids))


if __name__ == "__main__":
    unittest.main()
