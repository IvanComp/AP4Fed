
import sys
import os
import numpy as np

# Mocking task_handler if needed, but get_client_indices is in main.py
# Let's import directly from main.py if possible, or copy it for testing.

def test_get_client_indices():
    # Mocking the function from main.py for isolated test
    def get_client_indices_mock(dataset_len, cid, n_clients, config, current_round, total_rounds=10):
        total_samples = dataset_len
        indices = list(range(total_samples))
        
        persistence = config.get("data_persistence_type", "Same Data")
        
        if persistence == "New Data":
            frac = current_round / total_rounds
            subset_size = int(total_samples * min(frac, 1.0))
            indices = indices[:subset_size]
        elif persistence == "Remove Data":
            frac = (total_rounds - current_round + 1) / total_rounds
            subset_size = int(total_samples * max(frac, 0.1))
            indices = indices[:subset_size]

        dist_type = config.get("data_distribution_type", "IID")
        # Skipping Non-IID sort in mock for simplicity
        
        samples_per_client = len(indices) // n_clients
        if samples_per_client == 0: samples_per_client = 1
        
        start_idx = (int(cid) % n_clients) * samples_per_client
        end_idx = start_idx + samples_per_client
        return indices[start_idx:end_idx]

    dataset_len = 1000
    n_clients = 2
    total_rounds = 10

    # Test Same Data
    conf_same = {"data_persistence_type": "Same Data"}
    idx_r1 = get_client_indices_mock(dataset_len, 0, n_clients, conf_same, 1, total_rounds)
    idx_r10 = get_client_indices_mock(dataset_len, 0, n_clients, conf_same, 10, total_rounds)
    assert len(idx_r1) == len(idx_r10) == 500, f"Same Data failed: {len(idx_r1)}, {len(idx_r10)}"

    # Test New Data (Inflow)
    conf_new = {"data_persistence_type": "New Data"}
    idx_r1 = get_client_indices_mock(dataset_len, 0, n_clients, conf_new, 1, total_rounds)
    idx_r5 = get_client_indices_mock(dataset_len, 0, n_clients, conf_new, 5, total_rounds)
    idx_r10 = get_client_indices_mock(dataset_len, 0, n_clients, conf_new, 10, total_rounds)
    
    print(f"New Data - R1: {len(idx_r1)}, R5: {len(idx_r5)}, R10: {len(idx_r10)}")
    assert len(idx_r1) < len(idx_r5) < len(idx_r10), "New Data (Inflow) logic failed trend"
    assert len(idx_r1) == int(1000 * 0.1) // 2
    assert len(idx_r10) == 500

    # Test Remove Data
    conf_rem = {"data_persistence_type": "Remove Data"}
    idx_r1 = get_client_indices_mock(dataset_len, 0, n_clients, conf_rem, 1, total_rounds)
    idx_r5 = get_client_indices_mock(dataset_len, 0, n_clients, conf_rem, 5, total_rounds)
    idx_r10 = get_client_indices_mock(dataset_len, 0, n_clients, conf_rem, 10, total_rounds)
    
    print(f"Remove Data - R1: {len(idx_r1)}, R5: {len(idx_r5)}, R10: {len(idx_r10)}")
    assert len(idx_r1) > len(idx_r5) > len(idx_r10), "Remove Data logic failed trend"
    
    print("All persistence logic tests passed!")

if __name__ == "__main__":
    test_get_client_indices()
