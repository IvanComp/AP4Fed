import random
import unittest

import torch
from torch.utils.data import TensorDataset

from Docker.taskA import (
    apply_hdh_once,
    augment_agnews_text,
    build_client_partition_map,
    should_reload_data,
)


class DataPartitionTests(unittest.TestCase):
    def setUp(self):
        features = torch.arange(160, dtype=torch.float32).reshape(40, 4)
        labels = torch.tensor([label for label in range(4) for _ in range(10)])
        self.dataset = TensorDataset(features, labels)

    @staticmethod
    def clients(alpha):
        return [
            {
                "client_id": client_id,
                "dataset": "AG_NEWS",
                "data_distribution_type": "non-IID",
                "non_iid_alpha": alpha,
            }
            for client_id in range(1, 5)
        ]

    def test_partitions_are_disjoint_balanced_and_complete(self):
        partitions = build_client_partition_map(
            self.dataset, self.clients(0.1), "AG_NEWS", seed=1234
        )
        flattened = [idx for indices in partitions.values() for idx in indices]
        self.assertEqual(list(range(40)), sorted(flattened))
        self.assertEqual(40, len(set(flattened)))
        self.assertEqual([10, 10, 10, 10], [len(partitions[c]) for c in sorted(partitions)])

    def test_partition_is_deterministic_and_uses_alpha(self):
        low_a = build_client_partition_map(
            self.dataset, self.clients(0.1), "AG_NEWS", seed=99
        )
        low_b = build_client_partition_map(
            self.dataset, self.clients(0.1), "AG_NEWS", seed=99
        )
        high = build_client_partition_map(
            self.dataset, self.clients(1.0), "AG_NEWS", seed=99
        )
        self.assertEqual(low_a, low_b)
        self.assertNotEqual(low_a, high)

    def test_iid_clients_remain_class_balanced_in_a_mixed_partition(self):
        clients = [
            {
                "client_id": client_id,
                "dataset": "AG_NEWS",
                "data_distribution_type": "IID" if client_id <= 2 else "non-IID",
                "non_iid_alpha": 0.5,
            }
            for client_id in range(1, 5)
        ]
        partitions = build_client_partition_map(
            self.dataset, clients, "AG_NEWS", seed=1
        )
        for client_id in (1, 2):
            labels = [int(self.dataset[idx][1]) for idx in partitions[client_id]]
            counts = [labels.count(label) for label in range(4)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_same_data_is_reused_across_rounds(self):
        loader = object()
        self.assertTrue(should_reload_data(None, "Same Data", None, 1))
        self.assertFalse(should_reload_data(loader, "Same Data", 1, 2))
        self.assertTrue(should_reload_data(loader, "New Data", 1, 2))

    def test_hdh_runs_once_and_keeps_the_enriched_loader(self):
        initial_loader = object()
        enriched_loader = object()
        calls = []

        def rebalance(loader):
            calls.append(loader)
            return enriched_loader, 12.5

        loader, applied, hdh_ms = apply_hdh_once(
            initial_loader, True, "non-IID", False, rebalance
        )
        self.assertIs(loader, enriched_loader)
        self.assertTrue(applied)
        self.assertEqual(12.5, hdh_ms)

        loader, applied, hdh_ms = apply_hdh_once(
            loader, True, "non-IID", applied, rebalance
        )
        self.assertIs(loader, enriched_loader)
        self.assertTrue(applied)
        self.assertEqual(0.0, hdh_ms)
        self.assertEqual([initial_loader], calls)

    def test_hdh_never_runs_for_iid_clients(self):
        initial_loader = object()

        def unexpected_rebalance(_loader):
            self.fail("HDH must not run for an IID client")

        loader, applied, hdh_ms = apply_hdh_once(
            initial_loader, True, "IID", False, unexpected_rebalance
        )
        self.assertIs(loader, initial_loader)
        self.assertFalse(applied)
        self.assertEqual(0.0, hdh_ms)

    def test_agnews_augmentation_does_not_copy_the_source_text(self):
        source = "markets gain after technology earnings rise"
        augmented = augment_agnews_text(source, random.Random(1))
        self.assertNotEqual(source, augmented)
        self.assertTrue(augmented)


if __name__ == "__main__":
    unittest.main()
