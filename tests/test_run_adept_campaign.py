import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from run_adept_campaign import (
    CONFIGURATIONS,
    DOCKER_COMPOSE_PATH,
    MODELS,
    RunSpec,
    build_config,
    build_docker_compose,
    build_plan,
)


class CampaignPlanTests(unittest.TestCase):
    def setUp(self):
        self.counts = {
            ("OFF,OFF,OFF", "CNN 16k"): 5,
            ("OFF,OFF,OFF", "squeezenet1_1"): 5,
            ("ON,OFF,OFF", "CNN 16k"): 5,
            ("ON,OFF,OFF", "squeezenet1_1"): 5,
            ("OFF,ON,OFF", "CNN 16k"): 5,
            ("OFF,OFF,ON", "CNN 16k"): 5,
            ("OFF,OFF,ON", "squeezenet1_1"): 2,
        }

    def test_full_plan_has_48_missing_runs(self):
        plan = build_plan(self.counts, set(), target_repeats=5)
        self.assertEqual(48, len(plan))

    def test_first_wave_populates_all_nine_empty_cells(self):
        plan = build_plan(self.counts, set(), target_repeats=5, through_wave=1)
        self.assertEqual(9, len(plan))
        self.assertEqual({1}, {spec.repeat for spec in plan})
        self.assertEqual(len(plan), len({(spec.configuration, spec.model) for spec in plan}))

    def test_plan_is_breadth_first(self):
        plan = build_plan(self.counts, set(), target_repeats=5)
        repeats = [spec.repeat for spec in plan]
        self.assertEqual(repeats, sorted(repeats))
        self.assertEqual(9, repeats.count(1))
        self.assertEqual(9, repeats.count(2))
        self.assertEqual(10, repeats.count(3))

    def test_completed_campaign_slot_is_skipped(self):
        completed = {RunSpec(1, "squeezenet1_1", "OFF,ON,OFF").run_id}
        plan = build_plan(self.counts, completed, target_repeats=5, through_wave=1)
        self.assertEqual(8, len(plan))

    def test_config_matches_preliminary_experiment_profile(self):
        spec = RunSpec(1, "squeezenet1_1", "ON,ON,ON")
        config = build_config(spec, rounds=10)
        self.assertEqual(10, config["rounds"])
        self.assertEqual([3, 3, 3, 3, 1], [client["cpu"] for client in config["client_details"]])
        self.assertTrue(config["patterns"]["client_selector"]["enabled"])
        self.assertTrue(config["patterns"]["message_compressor"]["enabled"])
        self.assertTrue(config["patterns"]["heterogeneous_data_handler"]["enabled"])

    def test_docker_phase_starts_from_a_full_80_run_matrix(self):
        plan = build_plan({}, set(), target_repeats=5, execution_mode="Docker")
        self.assertEqual(80, len(plan))
        self.assertTrue(all(spec.mode == "Docker" for spec in plan))
        self.assertTrue(all("__docker__" in spec.run_id for spec in plan))

    def test_docker_config_and_compose_have_five_explicit_clients(self):
        spec = RunSpec(1, "CNN 16k", "ON,OFF,ON", execution_mode="Docker")
        config = build_config(spec, rounds=10)
        self.assertEqual("Docker", config["simulation_type"])
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "compose.yml"
            build_docker_compose(config, source=DOCKER_COMPOSE_PATH, destination=destination)
            compose = yaml.safe_load(destination.read_text(encoding="utf-8"))
        self.assertEqual(
            {"server", "client1", "client2", "client3", "client4", "client5"},
            set(compose["services"]),
        )
        self.assertEqual(1, compose["services"]["client5"]["cpus"])


if __name__ == "__main__":
    unittest.main()
