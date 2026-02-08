import os

# --- Disable Ray Metrics and Usage Stats to silence "RpcError: 14" ---
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_metrics_export_bin_path"] = ""
os.environ["RAY_ENABLE_METRICS_EXPORT"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

import json
import random
import time
import base64
import zlib
import pickle
import numpy as np
import torch
import psutil
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, FitRes
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy

import logging
import warnings
logging.getLogger("flwr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*dtype.*align.*") # Suppress torchvision/numpy warning
warnings.filterwarnings("ignore", message=".*Ray will no longer override.*") # Suppress Ray warning
warnings.filterwarnings("ignore", message=".*Failed to establish connection to the event+metrics exporter agent.*")
warnings.filterwarnings("ignore", message=".*Failed to establish connection to the metrics exporter agent.*")
warnings.filterwarnings("ignore", message=".*__array__ implementation doesn't accept a copy keyword.*")
warnings.filterwarnings("ignore", message=".*mode.*deprecated.*Pillow.*")

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

import task_handler

# --- Data Stores ---
GLOBAL_HISTORY = [] # List of per-client-per-round records

# --- Configuration Loader ---

def load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                return json.load(f)
            except:
                pass
    return {}

CONFIG = load_config()

# --- Global State ---

CLIENT_DETAILS = CONFIG.get("client_details", [])
PATTERNS = CONFIG.get("patterns", {})
ROUNDS = int(CONFIG.get("rounds", 5))
NUM_CLIENTS = int(CONFIG.get("clients", 2))

# --- Data Loading (Global for now) ---
# Note: On some systems, globals are not shared across processes. 
# We might need to reload in client_fn if simulation fails.
# For now, we follow the baseline pattern.

def load_global_data():
    default_client = CLIENT_DETAILS[0] if CLIENT_DETAILS else {}
    dataset_name = default_client.get("dataset", "CIFAR10")
    try:
        return task_handler.get_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, 10, 32, 3

TRAINSET_GLOBAL, TESTSET_GLOBAL, NUM_CLASSES, INPUT_SIZE, IN_CHANNELS = load_global_data()
GLOBAL_ROUND_COUNTER = 1

# --- Data Partitioning ---

def get_client_indices(dataset, cid, n_clients, config, current_round):
    if dataset is None: return []
    total_samples = len(dataset)
    indices = list(range(total_samples))
    
    # 5. Data Inflow (Persistence) Mechanism
    persistence = config.get("data_persistence_type", "Same Data")
    total_rounds = ROUNDS
    
    if persistence == "New Data":
        frac = current_round / total_rounds
        subset_size = int(total_samples * min(frac, 1.0))
        indices = indices[:subset_size]
    elif persistence == "Remove Data":
        frac = (total_rounds - current_round + 1) / total_rounds
        subset_size = int(total_samples * max(frac, 0.1))
        indices = indices[:subset_size]

    dist_type = config.get("data_distribution_type", "IID")
    if dist_type == "Non-IID":
        try:
            targets = []
            if hasattr(dataset, "targets"):
                targets = np.array(list(dataset.targets)) # list() avoids NumPy 2.0 copy keyword issue
            else:
                targets = np.array([y for _, y in dataset])
            # Filter targets to match current indices if indices were sliced
            # However, simpler to sort everything then slice, OR slice then sort.
            # AP4FED usually sorts once.
            indices.sort(key=lambda i: targets[i])
        except Exception as e:
            print(f"Non-IID Sort Error: {e}")

    # Basic partitioning
    samples_per_client = len(indices) // n_clients
    if samples_per_client == 0: samples_per_client = 1
    
    start_idx = (int(cid) % n_clients) * samples_per_client
    end_idx = start_idx + samples_per_client
    return indices[start_idx:end_idx]

# This line needs to be updated to call the new get_client_indices signature
# The original line was: CLIENT_INDICES = get_client_indices(TRAINSET_GLOBAL, NUM_CLIENTS, CLIENT_DETAILS)
# This global call won't work with the new signature which expects cid and config.
# This implies that client indices will now be determined within the client_fn or client class.
# For now, we'll keep a placeholder or remove it if it's no longer used globally.
# Given the new signature, CLIENT_INDICES cannot be pre-calculated globally like this.
# It will need to be calculated per client.
# For the purpose of this edit, I will comment out the old line and assume the new function
# will be called appropriately where client-specific indices are needed.
# CLIENT_INDICES = get_client_indices(TRAINSET_GLOBAL, NUM_CLIENTS, CLIENT_DETAILS)

# --- Custom Strategy (for Patterns) ---

class AP4FedStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        global GLOBAL_ROUND_COUNTER
        decoded_results = []
        for client, fit_res in results:
            cid = client.cid
            metrics = fit_res.metrics
            
            # PATTERN: Message Compressor (Decompression server-side?)
            if "compressed_params" in fit_res.metrics:
                try:
                    b64_str = fit_res.metrics["compressed_params"]
                    payload = base64.b64decode(b64_str)
                    decompressed = pickle.loads(zlib.decompress(payload))
                    fit_res.parameters = fl.common.ndarrays_to_parameters(decompressed)
                    del fit_res.metrics["compressed_params"]
                except Exception as e:
                    print(f"Server Decompression Error: {e}")
            
            decoded_results.append((client, fit_res))
            
            # Update global round counter for the NEXT round
            if server_round >= GLOBAL_ROUND_COUNTER:
                GLOBAL_ROUND_COUNTER = server_round + 1

            # Store partial record for GLOBAL_HISTORY
            GLOBAL_HISTORY.append({
                "round": server_round,
                "cid": cid,
                "type": "fit",
                "training_time": metrics.get("training_time", 0.0),
                "cpu_usage": metrics.get("cpu_usage", 0.0),
                "ram_usage": metrics.get("ram_usage", 0.0),
                "hdh_time": metrics.get("hdh_time", 0.0),
                "loss": metrics.get("loss", 0.0),
                "skipped": metrics.get("skipped", False),
                "cpu_config": metrics.get("cpu_config", 1),
                "dist_type": metrics.get("dist_type", "IID"),
                "client_name": metrics.get("client_name", f"Client {cid}"),
                "cluster": metrics.get("cluster", "Default"),
                "total_round_time": 0.0, # Placeholder, filled in aggregate_evaluate
                "comm_time": 0.0 # Placeholder
            })
            
        agg_params, metrics = super().aggregate_fit(server_round, decoded_results, failures)
        
        if PATTERNS.get("model_co-versioning_registry", {}).get("enabled", False) and agg_params is not None:
            try:
                os.makedirs("model_weights", exist_ok=True)
                path = f"model_weights/round_{server_round}.pt"
                # Save as parameters for mockup
                with open(path, "wb") as f:
                    pickle.dump(agg_params, f)
            except Exception as e:
                print(f"Registry Error: {e}")

        return agg_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        # Call super first to trigger tracker.stop_evaluate
        agg_res = super().aggregate_evaluate(server_round, results, failures)
        
        # Get round duration from tracker
        round_duration = 0.0
        if hasattr(self, 'tracker') and len(self.tracker.durations) >= server_round:
            round_duration = self.tracker.durations[server_round-1]

        for client, eval_res in results:
            cid = client.cid
            metrics = eval_res.metrics
            # Update history with eval metrics and timing
            for record in GLOBAL_HISTORY:
                # cid from ClientProxy is often a string, record["cid"] might be int or str
                if record["round"] == server_round and str(record["cid"]) == str(cid):
                    record["val_loss"] = eval_res.loss
                    record["val_acc"] = metrics.get("accuracy", 0.0)
                    record["val_f1"] = metrics.get("f1", 0.0)
                    record["val_mae"] = metrics.get("mae", 0.0)
                    
                    record["total_round_time"] = round_duration
                    # Comm Time = Total - (Train + HDH)
                    overhead = record.get("training_time", 0) + record.get("hdh_time", 0)
                    record["comm_time"] = max(0, round_duration - overhead)
                    break
        return agg_res

# --- Flower Client ---


def get_loader(dataset, batch_size, shuffle=True):
    # Check for custom collate_fn attached to the dataset (or its parent if Subset)
    collate_fn = None
    if hasattr(dataset, "custom_collate_fn"):
        collate_fn = dataset.custom_collate_fn
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "custom_collate_fn"):
         collate_fn = dataset.dataset.custom_collate_fn
         
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, client_config, trainset, testset, device):
        self.cid = cid
        self.config = client_config
        self.trainset = trainset
        self.testset = testset
        self.device = device
        
        self.model_name = self.config.get("model", "CNN 16k")
        self.dataset_name = self.config.get("dataset", "CIFAR10")
        
        # Initialize model
        self.model = task_handler.get_model(
            self.model_name, 
            self.dataset_name, 
            NUM_CLASSES, 
            INPUT_SIZE, 
            IN_CHANNELS
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update round from server config if available
        current_round = config.get("server_round", 1)
        # Update indices based on current round (Inflow/Outflow support)
        my_indices = get_client_indices(TRAINSET_GLOBAL, self.cid - 1, NUM_CLIENTS, self.config, current_round)
        self.trainset = Subset(TRAINSET_GLOBAL, my_indices)

        # --- Metrics Collection Start ---
        t_start = time.time()
        cpu_start = psutil.cpu_percent(interval=None)
        ram_start = psutil.virtual_memory().percent
        
        # --- PATTERNS (Selector, HDH, Registry, etc.) ---
        
        # 1. client_selector
        if PATTERNS.get("client_selector", {}).get("enabled", False):
            params = PATTERNS["client_selector"].get("params", {})
            criteria = params.get("selection_criteria", "CPU").lower()
            threshold = int(params.get("selection_value", 0))
            if int(self.config.get(criteria, 1)) < threshold:
                return self.get_parameters(config={}), len(self.trainset), {"skipped": True, "client_name": f"Client {self.cid}"}

        # 2. client_cluster
        cluster_grp = "Default"
        if PATTERNS.get("client_cluster", {}).get("enabled", False):
            params = PATTERNS["client_cluster"].get("params", {})
            criteria = params.get("clustering_criteria", "CPU").lower()
            threshold = int(params.get("selection_value", 2))
            cluster_grp = "High-Res" if int(self.config.get(criteria, 1)) >= threshold else "Low-Res"

        # 3. heterogeneous_data_handler (rebalancing)
        hdh_time = 0.0
        if PATTERNS.get("heterogeneous_data_handler", {}).get("enabled", False):
            # Check if text or image
            t_hdh_start = time.time()
            self.trainset = task_handler.apply_hdh_gan(self.trainset, num_classes=10, device=self.device)
            hdh_time = time.time() - t_hdh_start

        # 5. Network Stability / Delay Injection
        if self.config.get("delay_combobox") == "Yes":
            time.sleep(0.05) # 50ms delay

        self.set_parameters(parameters)
        
        # Prepare DataLoader
        trainloader = get_loader(self.trainset, batch_size=32, shuffle=True)
        
        # 4. multi-task_model_trainer
        epochs = int(self.config.get("epochs", 1))
        if PATTERNS.get("multi-task_model_trainer", {}).get("enabled", False):
            epochs *= 2 # Simulate multi-task complexity
        
        # Train
        loss = task_handler.train(self.model, trainloader, epochs, self.device)
        
        # Metrics
        t_end = time.time()
        cpu_end = psutil.cpu_percent(interval=None)
        ram_end = psutil.virtual_memory().percent
        
        metrics = {
            "loss": loss,
            "training_time": t_end - t_start,
            "cpu_usage": (cpu_start + cpu_end) / 2,
            "ram_usage": (ram_start + ram_end) / 2,
            "hdh_time": hdh_time,
            "cpu_config": self.config.get("cpu", 1),
            "dist_type": self.config.get("data_distribution_type", "IID"),
            "client_name": f"Client {self.cid}",
            "cluster": cluster_grp
        }
        
        new_params = self.get_parameters(config={})
        
        # --- PATTERN: Message Compressor ---
        if PATTERNS.get("message_compressor", {}).get("enabled", False):
            serialized = pickle.dumps(new_params)
            compressed = zlib.compress(serialized)
            b64_str = base64.b64encode(compressed).decode('ascii')
            metrics["compressed_params"] = b64_str
            return [], len(self.trainset), metrics
        
        return new_params, len(self.trainset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Update round from server config
        current_round = config.get("server_round", 1)
        # Usually evaluation is on full set, but we could synchronize if needed.
        # For now, just ensuring consistency.
        valloader = get_loader(self.testset, batch_size=32, shuffle=False)
        loss, acc, f1, mae = task_handler.test(self.model, valloader, self.device)
        return float(loss), len(self.testset), {
            "accuracy": float(acc), 
            "f1": float(f1), 
            "mae": float(mae),
            "client_name": f"Client {self.cid}"
        }


# --- Main Orchestration ---


# --- 3. Performance Tracker ---

class PerformanceTracker:
    def __init__(self):
        self.fit_start_time = 0
        self.durations = []

    def start_fit(self, server_round: int):
        self.fit_start_time = time.time()
        return {"server_round": server_round}

    def start_evaluate(self, server_round: int):
        return {"server_round": server_round}

    def stop_evaluate(self, metrics: List[Tuple[int, Dict]]):
        if self.fit_start_time > 0:
            duration = time.time() - self.fit_start_time
            self.durations.append(duration)
        
        if not metrics:
            return {"accuracy": 0.0}
            
        accs = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        total_examples = sum(examples)
        return {"accuracy": sum(accs) / total_examples if total_examples > 0 else 0.0}

def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    # Aggregate fit metrics (e.g. loss)
    total_examples = sum([num_examples for num_examples, _ in metrics])
    avg_loss = sum([num_examples * m.get("loss", 0.0) for num_examples, m in metrics]) / total_examples if total_examples > 0 else 0.0
    
    # Pattern specific aggregations
    skipped_count = sum([1 for _, m in metrics if m.get("skipped", False)])
    
    return {
        "loss": avg_loss,
        "clients_skipped": skipped_count
    }

# --- Main Orchestration ---

def client_fn(context: Context):
    global GLOBAL_ROUND_COUNTER
    try:
        partition_id = int(context.node_config["partition-id"])
    except:
        partition_id = int(context) if isinstance(context, str) else 0
        
    if CLIENT_DETAILS:
        c_conf = CLIENT_DETAILS[partition_id % len(CLIENT_DETAILS)]
    else:
        c_conf = {"dataset": "CIFAR10", "model": "CNN 16k", "cpu": 1, "epochs": 1}
    
    current_round = GLOBAL_ROUND_COUNTER
    
    my_indices = get_client_indices(TRAINSET_GLOBAL, partition_id, NUM_CLIENTS, c_conf, current_round)
    my_trainset = Subset(TRAINSET_GLOBAL, my_indices)
    
    return FlowerClient(
        cid=partition_id + 1, 
        client_config=c_conf,
        trainset=my_trainset,
        testset=TESTSET_GLOBAL,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ).to_client()

def main():
    print("\n" + "="*50)
    print("      LIGHT-AP4FED SIMUATION STARTING")
    print("="*50)
    print(f"Clients:  {NUM_CLIENTS}")
    print(f"Rounds:   {ROUNDS}")
    
    print("\nClient Configuration Overview:")
    header = f"{'ID':<4} | {'Model':<10} | {'Dataset':<10} | {'Dist.':<8} | {'Persist.':<10} | {'Delay':<6} | {'CPU':<4}"
    print(header)
    print("-" * len(header))
    
    for i in range(NUM_CLIENTS):
        c = CLIENT_DETAILS[i] if i < len(CLIENT_DETAILS) else {}
        cid = i + 1
        model = c.get("model", "CNN 16k")
        dataset = c.get("dataset", "CIFAR10")
        dist = c.get("data_distribution_type", "IID")
        persistence = c.get("data_persistence_type", "Same Data")
        delay = c.get("delay_combobox", "No")
        cpu = c.get("cpu", 1)
        print(f"{cid:<4} | {model:<10} | {dataset:<10} | {dist:<8} | {persistence:<10} | {delay:<6} | {cpu:<4}")

    print("\nEnabled Architectural Patterns:")
    any_pattern = False
    for p, cfg in PATTERNS.items():
        if cfg.get("enabled", False):
            any_pattern = True
            line = f"- {p.replace('_', ' ').title()}"
            params = cfg.get("params", {})
            if p == "client_selector" and params:
                line += f" ( selection criteria = {params.get('selection_criteria', 'N/A')} threshold = {params.get('selection_value', '0')} )"
            elif p == "client_cluster" and params:
                line += f" ( clustering criteria = {params.get('clustering_criteria', 'N/A')} threshold = {params.get('selection_value', '0')} )"
            print(line)
    if not any_pattern:
        print("- None")
    
    print("-" * 50)
    
    if TRAINSET_GLOBAL is None:
        print("Error: Dataset not loaded.")
        return

    os.makedirs("csv", exist_ok=True)
    tracker = PerformanceTracker()

    # Strategy
    strategy = AP4FedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=tracker.start_fit,
        on_evaluate_config_fn=tracker.start_evaluate,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=tracker.stop_evaluate,
    )
    strategy.tracker = tracker # Inject tracker into strategy for timing access
    
    # Resources
    client_resources = {"num_cpus": 1.0}
    if torch.cuda.is_available():
        client_resources["num_gpus"] = 0.5

    # Start
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": psutil.cpu_count(),
        "configure_logging": True,
        "logging_level": logging.ERROR,
        "log_to_driver": True,
        "_system_config": {
            # Removed invalid parameters that caused GCS crash
        },
        "runtime_env": {
            "env_vars": {
                "RAY_USAGE_STATS_ENABLED": "0",
                "RAY_ENABLE_METRICS_EXPORT": "0",
                "RAY_metrics_export_bin_path": "",
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                "RAY_DEDUP_LOGS": "0"
            }
        }
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args
    )

    # Print Round Summary
    print("\n--- Simulation Summary ---")
    acc_history = history.metrics_distributed.get("accuracy", [])
    for rnd, acc in acc_history:
        print(f"Round {rnd}: Accuracy = {acc:.4f}")

    # Save Results to CSV (AP4FED 22-column format)
    import csv
    dataset_name = CLIENT_DETAILS[0].get("dataset", "Unknown") if CLIENT_DETAILS else "Unknown"
    model_name = CLIENT_DETAILS[0].get("model", "Unknown") if CLIENT_DETAILS else "Unknown"
    distr_type = CLIENT_DETAILS[0].get("data_distribution_type", "Unknown") if CLIENT_DETAILS else "Unknown"
    cpu_per_client = CLIENT_DETAILS[0].get("cpu", 1) if CLIENT_DETAILS else 1
    
    enabled_patterns = [p for p, cfg in PATTERNS.items() if cfg.get("enabled", False)]
    patterns_str = "|".join(enabled_patterns)

    timestamp = int(time.time())
    csv_path = f'csv/results_{dataset_name}_{model_name}_{timestamp}.csv'
    
    print(f"Saving per-client results to {csv_path}...")
    
    # Sort history by Round then Client ID (Numerical)
    try:
        GLOBAL_HISTORY.sort(key=lambda x: (int(x['round']), int(x['cid'])))
    except:
        GLOBAL_HISTORY.sort(key=lambda x: (x['round'], str(x['cid'])))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header (22 columns)
        writer.writerow([
            'Client ID', 'FL Round', 'Training Time', 'JSD', 'HDH Time', 
            'Communication Time', 'Total Time of FL Round', '# of CPU', 
            'CPU Usage (%)', 'RAM Usage (%)', 'Model Type', 'Data Distr. Type', 
            'Dataset', 'Train Loss', 'Train Accuracy', 'Train F1', 'Train MAE', 
            'Val Loss', 'Val Accuracy', 'Val F1', 'Val MAE', 'AP List'
        ])
        
        for record in GLOBAL_HISTORY:
            cid_str = record.get("client_name", f"Client {record['cid']}")
            rnd = record["round"]
            
            if record.get("skipped", False):
                # Merged row for excluded clients
                row = [cid_str, rnd] + ["Client excluded"] * 19 + ["-"]
            else:
                row = [
                    cid_str,
                    rnd,
                    f"{record.get('training_time', 0.0):.2f}",
                    "0.0", # JSD placeholder
                    f"{record.get('hdh_time', 0.0):.2f}",
                    f"{record.get('comm_time', 0.0):.2f}", 
                    f"{record.get('total_round_time', 0.0):.2f}",
                    record.get("cpu_config", 1), # Per-client CPU
                    f"{record.get('cpu_usage', 0.0):.2f}",
                    f"{record.get('ram_usage', 0.0):.2f}",
                    model_name,
                    record.get("dist_type", "IID"), # Per-client Distribution
                    dataset_name,
                    f"{record.get('loss', 0.0):.4f}",
                    "0.0", # Train Acc
                    "0.0", # Train F1
                    "0.0", # Train MAE
                    f"{record.get('val_loss', 0.0):.4f}",
                    f"{record.get('val_acc', 0.0):.4f}",
                    f"{record.get('val_f1', 0.0):.4f}",
                    f"{record.get('val_mae', 0.0):.4f}",
                    "|".join(enabled_patterns) + (f" (Cluster: {record.get('cluster', 'N/A')})" if "client_cluster" in enabled_patterns else "")
                ]
            writer.writerow(row)

    print("Simulation Completed Successfully.")

if __name__ == "__main__":
    main()
