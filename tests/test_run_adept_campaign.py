import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
import pandas as pd

from run_adept_campaign import (
    CONFIGURATIONS,
    DOCKER_COMPOSE_PATH,
    RunSpec,
    build_config,
    build_docker_compose,
    build_partition_seed,
    build_plan,
    build_singularity_step_command,
    configure_resource_profile,
    _terminate_processes,
    split_pattern_column,
    validate_complete_matrix,
    verified_successful_run_ids,
)


class FakeProcess:
    def __init__(self, running=True):
        self.returncode = None if running else 0
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def make_spec(**overrides):
    values = {
        "repeat": 1,
        "dataset": "AG_NEWS",
        "model": "MLP",
        "client_count": 4,
        "high_spec_percentage": 25,
        "non_iid_percentage": 25,
        "alpha": 0.5,
        "delay_percentage": 25,
        "configuration": "OFF,OFF,OFF",
        "execution_mode": "Docker",
    }
    values.update(overrides)
    return RunSpec(**values)


class CampaignPlanTests(unittest.TestCase):
    def test_pattern_columns_fall_back_to_campaign_index_configuration(self):
        frame = pd.DataFrame({"Final Val F1": [0.75]})
        result = split_pattern_column(frame, fallback_config_id="ON,OFF,OFF")
        self.assertEqual("ON,OFF,OFF", result.loc[0, "config_id"])
        self.assertEqual("ON", result.loc[0, "client_selector_pattern"])
        self.assertEqual("OFF", result.loc[0, "message_compressor_pattern"])
        self.assertEqual("OFF", result.loc[0, "hdh_pattern"])

    def test_singularity_cleanup_allows_graceful_client_exit(self):
        process = FakeProcess()
        _terminate_processes([process], graceful_timeout=1)
        self.assertEqual(0, process.returncode)
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)

    def test_campaign_has_600_docker_runs_for_ten_seeds(self):
        plan = build_plan(set(), target_repeats=10)
        self.assertEqual(600, len(plan))
        validate_complete_matrix(plan, 10)
        for repeat in range(1, 11):
            self.assertEqual(60, sum(spec.repeat == repeat for spec in plan))

    def test_complete_first_repetition_precedes_second(self):
        plan = build_plan(set(), target_repeats=10)
        self.assertTrue(all(spec.repeat == 1 for spec in plan[:60]))
        self.assertTrue(all(spec.repeat == 2 for spec in plan[60:120]))
        self.assertEqual(list(range(1, 11)), [plan[offset].repeat for offset in range(0, 600, 60)])

    def test_one_wave_shard_contains_exactly_60_runs(self):
        plan = build_plan(set(), target_repeats=10, only_wave=7)
        self.assertEqual(60, len(plan))
        self.assertEqual({7}, {spec.repeat for spec in plan})

    def test_112_core_profile_fits_largest_configuration(self):
        try:
            configure_resource_profile(112, 2, 7, 12)
            config = build_config(
                make_spec(client_count=10, high_spec_percentage=75), rounds=20
            )
            self.assertEqual(110, sum(client["cpu"] for client in config["client_details"]))
        finally:
            configure_resource_profile(32, 2, 2, 3)

    def test_only_requested_pattern_states_are_scheduled(self):
        plan = build_plan(set(), target_repeats=1)
        self.assertEqual(set(CONFIGURATIONS), {spec.configuration for spec in plan})

    def test_completed_run_is_skipped(self):
        complete = build_plan(set(), target_repeats=1)[0].run_id
        plan = build_plan({complete}, target_repeats=1)
        self.assertNotIn(complete, {spec.run_id for spec in plan})
        self.assertEqual(59, len(plan))

    def test_each_pattern_varies_only_its_stress_factors(self):
        plan = build_plan(set(), target_repeats=1, tasks=(("AG_NEWS", "MLP"),), client_counts=(4,))
        by_configuration = {}
        for spec in plan:
            by_configuration.setdefault(spec.configuration, []).append(spec)

        baseline = by_configuration["OFF,OFF,OFF"]
        self.assertEqual(1, len(baseline))
        self.assertEqual((75, 25, 0.5, 25), (
            baseline[0].high_spec_percentage,
            baseline[0].non_iid_percentage,
            baseline[0].alpha,
            baseline[0].delay_percentage,
        ))

        selector = by_configuration["ON,OFF,OFF"]
        self.assertEqual(
            [(25, 75), (50, 50), (75, 25)],
            [(spec.low_spec_percentage, spec.high_spec_percentage) for spec in selector],
        )
        self.assertEqual({(25, 0.5, 25)}, {
            (spec.non_iid_percentage, spec.alpha, spec.delay_percentage) for spec in selector
        })

        hdh = by_configuration["OFF,OFF,ON"]
        self.assertEqual(3, len(hdh))
        self.assertEqual({25, 50, 75}, {spec.non_iid_percentage for spec in hdh})
        self.assertEqual({0.5}, {spec.alpha for spec in hdh})
        self.assertEqual({(75, 25)}, {
            (spec.high_spec_percentage, spec.delay_percentage) for spec in hdh
        })

        compressor = by_configuration["OFF,ON,OFF"]
        self.assertEqual({25, 50, 75}, {spec.delay_percentage for spec in compressor})
        self.assertEqual({(75, 25, 0.5)}, {
            (spec.high_spec_percentage, spec.non_iid_percentage, spec.alpha)
            for spec in compressor
        })

    def test_config_encodes_stress_factors(self):
        spec = make_spec(
            client_count=10,
            high_spec_percentage=25,
            non_iid_percentage=50,
            alpha=0.5,
            delay_percentage=75,
            configuration="ON,OFF,OFF",
        )
        config = build_config(spec, rounds=20)
        clients = config["client_details"]
        self.assertEqual(20, config["rounds"])
        self.assertEqual(10, len(clients))
        self.assertEqual(2, sum(client["cpu"] == 3 for client in clients))
        self.assertEqual(8, sum(client["cpu"] == 2 for client in clients))
        self.assertEqual(20.0, config["campaign_metadata"]["realized_high_spec_percentage"])
        self.assertEqual(80.0, config["campaign_metadata"]["realized_low_spec_percentage"])
        self.assertTrue(all(client["ram"] == 4 for client in clients))
        self.assertEqual(5, sum(client["data_distribution_type"] == "non-IID" for client in clients))
        self.assertEqual(5, sum(client["data_distribution_type"] == "IID" for client in clients))
        self.assertTrue(
            all(
                client["non_iid_alpha"] == 0.5
                for client in clients
                if client["data_distribution_type"] == "non-IID"
            )
        )
        self.assertTrue(config["patterns"]["client_selector"]["enabled"])
        self.assertEqual(2, config["patterns"]["client_selector"]["params"]["selection_value"])
        realized = config["campaign_metadata"]["realized_delay_percentage"]
        self.assertEqual(80.0, realized)
        delayed = [client for client in clients if client["delay_combobox"] == "Yes"]
        not_delayed = [client for client in clients if client["delay_combobox"] == "No"]
        self.assertTrue(delayed)
        self.assertTrue(
            all(
                (client["delay_min_seconds"], client["delay_max_seconds"]) == (5, 10)
                for client in delayed
            )
        )
        self.assertTrue(
            all(
                (client["delay_min_seconds"], client["delay_max_seconds"]) == (0, 0)
                for client in not_delayed
            )
        )

    def test_pattern_labels_map_to_ap4fed_order(self):
        expected = {
            "OFF,OFF,OFF": (False, False, False),
            "ON,OFF,OFF": (True, False, False),
            "OFF,ON,OFF": (False, True, False),
            "OFF,OFF,ON": (False, False, True),
        }
        for configuration, states in expected.items():
            patterns = build_config(make_spec(configuration=configuration), rounds=20)["patterns"]
            actual = (
                patterns["client_selector"]["enabled"],
                patterns["message_compressor"]["enabled"],
                patterns["heterogeneous_data_handler"]["enabled"],
            )
            self.assertEqual(states, actual)

    def test_matched_pattern_runs_share_partition_seed(self):
        baseline = make_spec(configuration="OFF,OFF,OFF")
        hdh = make_spec(configuration="OFF,OFF,ON")
        self.assertEqual(build_partition_seed(baseline, 20), build_partition_seed(hdh, 20))
        self.assertNotEqual(
            build_partition_seed(baseline, 20),
            build_partition_seed(make_spec(repeat=2), 20),
        )
        self.assertEqual(
            list(range(1, 11)),
            [build_partition_seed(make_spec(repeat=repeat), 20) for repeat in range(1, 11)],
        )

    def test_percentage_levels_form_nested_client_sets(self):
        configs = {
            percentage: build_config(
                make_spec(
                    client_count=8,
                    high_spec_percentage=percentage,
                    non_iid_percentage=percentage,
                    delay_percentage=percentage,
                ),
                rounds=20,
            )
            for percentage in (25, 50, 75)
        }
        extractors = (
            lambda client: client["cpu"] == 3,
            lambda client: client["data_distribution_type"] == "non-IID",
            lambda client: client["delay_combobox"] == "Yes",
        )
        for selected in extractors:
            sets = {
                percentage: {
                    client["client_id"]
                    for client in config["client_details"]
                    if selected(client)
                }
                for percentage, config in configs.items()
            }
            self.assertLessEqual(sets[25], sets[50])
            self.assertLessEqual(sets[50], sets[75])

    def test_docker_compose_has_one_service_per_client(self):
        config = build_config(make_spec(client_count=8, execution_mode="Docker"), rounds=20)
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "compose.yml"
            build_docker_compose(config, source=DOCKER_COMPOSE_PATH, destination=destination)
            compose = yaml.safe_load(destination.read_text(encoding="utf-8"))
        self.assertEqual(9, len(compose["services"]))
        self.assertEqual("4g", compose["services"]["client1"]["mem_limit"])

    def test_worst_case_fits_32_cores_without_cpuset_overlap(self):
        config = build_config(
            make_spec(
                client_count=10,
                high_spec_percentage=75,
                configuration="ON,OFF,OFF",
            ),
            rounds=20,
        )
        self.assertEqual(28, sum(client["cpu"] for client in config["client_details"]))
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "compose.yml"
            build_docker_compose(config, source=DOCKER_COMPOSE_PATH, destination=destination)
            services = yaml.safe_load(destination.read_text(encoding="utf-8"))["services"]

        assigned = []
        for service in services.values():
            assigned.extend(int(cpu) for cpu in str(service["cpuset"]).split(","))
        self.assertEqual(30, len(assigned))
        self.assertEqual(30, len(set(assigned)))
        self.assertEqual(set(range(30)), set(assigned))

    def test_singularity_step_preserves_client_resources_and_environment(self):
        command = build_singularity_step_command(
            "/usr/bin/singularity",
            Path("/work/ap4fed.sif"),
            cpus=2,
            memory_gb=4,
            environment={"CLIENT_ID": "3", "SERVER_ADDRESS": "127.0.0.1:8080"},
            program="client.py",
        )
        self.assertIn("--cpus-per-task=2", command)
        self.assertIn("--mem=4G", command)
        self.assertIn("CLIENT_ID=3", command)
        self.assertIn("SERVER_ADDRESS=127.0.0.1:8080", command)
        self.assertEqual(["python", "client.py"], command[-2:])

    def test_resume_requires_archived_config_and_summary(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            summary = run_dir / "summary.csv"
            summary.write_text("a\n1\n", encoding="utf-8")
            index = root / "index.csv"
            with index.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Run ID", "Status", "Output Dir", "ML Summary CSV"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Run ID": "complete",
                        "Status": "ok",
                        "Output Dir": run_dir,
                        "ML Summary CSV": summary,
                    }
                )
                writer.writerow(
                    {
                        "Run ID": "missing",
                        "Status": "ok",
                        "Output Dir": root / "missing",
                        "ML Summary CSV": root / "missing.csv",
                    }
                )
                writer.writerow(
                    {
                        "Run ID": "interrupted",
                        "Status": "interrupted",
                        "Output Dir": run_dir,
                        "ML Summary CSV": summary,
                    }
                )
            self.assertEqual({"complete"}, verified_successful_run_ids(index))


if __name__ == "__main__":
    unittest.main()
