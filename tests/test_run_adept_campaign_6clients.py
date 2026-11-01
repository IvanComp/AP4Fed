import subprocess
import sys
import unittest


class SixClientCampaignTests(unittest.TestCase):
    def dry_run(self, campaign):
        return subprocess.run(
            [
                sys.executable,
                "run_adept_campaign_6clients.py",
                "--campaign",
                campaign,
                "--dry-run",
                "--target-repeats",
                "1",
                "--skip-dataset-prefetch",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_standard_matrix_has_46_six_client_runs_per_wave(self):
        output = self.dry_run("single")
        self.assertIn("Docker phase: 46 scheduled run(s)", output)
        self.assertIn("__n6__", output)

    def test_combined_matrix_has_32_six_client_runs_per_wave(self):
        output = self.dry_run("combinedAP")
        self.assertIn("Docker phase: 32 scheduled run(s)", output)
        self.assertIn("__n6__", output)


if __name__ == "__main__":
    unittest.main()
