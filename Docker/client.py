from multiprocessing import Process
import json
import os
import torch
import platform
import time
import zlib
import pickle
import numpy as np
import psutil
import socket
from multiprocessing import Process
from datetime import datetime
from io import BytesIO
from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context
from flwr.common.logger import log
from logging import INFO
from APClient import ClientRegistry
from taskA import (
    Net as NetA,
    get_weights as get_weights_A,
    load_data as load_data_A,
    set_weights as set_weights_A,
    train as train_A,
    test as test_A
)

global_client_details = None

def load_client_details():
    global global_client_details
    if global_client_details is None:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        config_dir = os.path.join(current_dir, 'configuration')
        config_file = os.path.join(config_dir, 'config.json')
        with open(config_file, 'r') as f:
            configJSON = json.load(f)
        client_details_list = configJSON.get("client_details", [])
        def client_sort_key(item):
            try:
                return int(item.get("client_id", 0))
            except:
                return 0
        client_details_list = sorted(client_details_list, key=client_sort_key)
        global_client_details = client_details_list
    return global_client_details

class ConfigServer:
    def __init__(self, config_list):
        self.config_list = config_list
        self.counter = 0
        self.assignments = {}

    def get_config_for(self, client_id: str):
        if client_id in self.assignments:
            return self.assignments[client_id]
        if self.counter < len(self.config_list):
            config = self.config_list[self.counter]
            self.assignments[client_id] = config
            self.counter += 1
            return config

CLIENT_REGISTRY = ClientRegistry()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1 

def set_cpu_affinity(process_pid: int, num_cpus: int) -> bool:
    try:
        process = psutil.Process(process_pid)
        total_cpus = os.cpu_count() or 1
        cpus_to_use = min(num_cpus, total_cpus)

        if cpus_to_use <= 0:
            return False

        target_cpus = list(range(cpus_to_use))
        if platform.system() in ("Linux", "Windows"):
            process.cpu_affinity(target_cpus)
        else:
            process.nice(10)
        return True
    except Exception:
        return False

# Lettura uso memoria nel container (cgroup)
def get_ram_percent_cgroup():
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            used = int(f.read())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit = int(f.read())
        return used / limit * 100
    except Exception:
        return psutil.Process(os.getpid()).memory_percent()

# Lettura uso CPU nel container (cgroup)
def get_cpu_percent_cgroup(interval: float = 1.0) -> float:
    try:
        with open("/sys/fs/cgroup/cpu/cpuacct.usage") as f:
            start = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        time.sleep(interval)
        with open("/sys/fs/cgroup/cpu/cpuacct.usage") as f:
            end = int(f.read())
        delta_ns = end - start
        elapsed_ns = interval * 1e9
        cores = (quota / period) if quota > 0 else (os.cpu_count() or 1)
        return (delta_ns / elapsed_ns / cores) * 100
    except Exception:
        return psutil.cpu_percent(interval=interval)

class FlowerClient(NumPyClient):
    def __init__(self, client_config: dict, model_type: str):
        self.client_config = client_config
        hostname = socket.gethostname()
        self.cid = hostname if hostname else client_config.get("client_id", "unknown")
        self.n_cpu = client_config.get("cpu")
        self.ram = client_config.get("ram")
        self.dataset = client_config.get("dataset")
        self.data_distribution_type = client_config.get("data_distribution_type")
        self.model = client_config.get("model")
        self.model_type = model_type

        current_os = platform.system()

        if self.n_cpu is not None:
            try:
                num_cpus_int = int(self.n_cpu)
                if num_cpus_int > 0:
                    set_cpu_affinity(os.getpid(), num_cpus_int)
            except Exception:
                pass

        CLIENT_REGISTRY.register_client(self.cid, model_type)
        self.net = NetA().to(DEVICE)
        self.trainloader, self.testloader = load_data_A(self.client_config)
        self.DEVICE = DEVICE

    def fit(self, parameters, config):
        global GLOBAL_ROUND_COUNTER
        proc = psutil.Process(os.getpid())
        cpu_start = proc.cpu_times().user + proc.cpu_times().system
        wall_start = time.time()

        compressed_parameters_hex = config.get("compressed_parameters_hex")
        global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MODEL_COVERSIONING, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
        CLIENT_SELECTOR = False
        CLIENT_CLUSTER = False
        MESSAGE_COMPRESSOR = False
        MODEL_COVERSIONING = False
        MULTI_TASK_MODEL_TRAINER = False
        HETEROGENEOUS_DATA_HANDLER = False

        current_dir = os.path.abspath(os.path.dirname(__file__))
        config_dir = os.path.join(current_dir, '..', 'configuration')
        config_file = os.path.join(config_dir, 'config.json')

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                configJSON = json.load(f)
            for pattern_name, pattern_info in configJSON.get("patterns", {}).items():
                if pattern_info.get("enabled"):
                    if pattern_name == "client_selector": CLIENT_SELECTOR = True
                    elif pattern_name == "client_cluster": CLIENT_CLUSTER = True
                    elif pattern_name == "message_compressor": MESSAGE_COMPRESSOR = True
                    elif pattern_name == "model_co-versioning_registry": MODEL_COVERSIONING = True
                    elif pattern_name == "multi-task_model_trainer": MULTI_TASK_MODEL_TRAINER = True
                    elif pattern_name == "heterogeneous_data_handler": HETEROGENEOUS_DATA_HANDLER = True

        n_cpu = self.n_cpu
        ram = self.ram
        dataset = self.dataset
        data_distribution_type = self.data_distribution_type

        if CLIENT_SELECTOR:
            selector_params = configJSON["patterns"]["client_selector"]["params"]
            selection_strategy = selector_params.get("selection_strategy", "")
            selection_criteria = selector_params.get("selection_criteria", "")
            selection_value = selector_params.get("selection_value", 0)
            if selection_strategy == "Resource-Based":
                if selection_criteria == "CPU" and n_cpu < selection_value:
                    log(INFO, f"Client {self.cid} has insufficient CPU ({n_cpu}). Will not participate.")
                    return parameters, 0, {}
                if selection_criteria == "RAM" and ram < selection_value:
                    log(INFO, f"Client {self.cid} has insufficient RAM ({ram}). Will not participate.")
                    return parameters, 0, {}
            log(INFO, f"Client {self.cid} participates in this round. (CPU: {n_cpu}, RAM: {ram})")

        if MESSAGE_COMPRESSOR and compressed_parameters_hex:
            comp = bytes.fromhex(compressed_parameters_hex)
            decompressed = pickle.loads(zlib.decompress(comp))
            numpy_arrays = [np.load(BytesIO(t)) for t in decompressed.tensors]
            parameters = [arr.astype(np.float32) for arr in numpy_arrays]

        set_weights_A(self.net, parameters)
        results, training_time = train_A(
            self.net, self.trainloader, self.testloader,
            epochs=1, DEVICE=self.DEVICE
        )
        communication_time = time.time()

        new_parameters = get_weights_A(self.net)

        round_number = GLOBAL_ROUND_COUNTER
        GLOBAL_ROUND_COUNTER += 1

        wall_end = time.time()
        cpu_end = proc.cpu_times().user + proc.cpu_times().system
        cpu_percent = get_cpu_percent_cgroup()
        ram_percent = get_ram_percent_cgroup()

        if MODEL_COVERSIONING:
            client_folder = os.path.join("model_weights", "clients", str(self.cid))
            os.makedirs(client_folder, exist_ok=True)
            client_file_path = os.path.join(client_folder, f"MW_round{round_number}.pt")
            torch.save(self.net.state_dict(), client_file_path)
            log(INFO, f"Client {self.cid} model weights saved to {client_file_path}")

        if MESSAGE_COMPRESSOR:
            serialized_parameters = pickle.dumps(new_parameters)
            original_size = len(serialized_parameters)
            compressed_parameters = zlib.compress(serialized_parameters)
            compressed_parameters_hex = compressed_parameters.hex()
            reduction_bytes = original_size - len(compressed_parameters)
            reduction_percentage = (reduction_bytes / original_size) * 100
            log(INFO, f"Local parameters compressed: reduced {reduction_bytes} bytes ({reduction_percentage:.2f}%)")
            metrics = {
                "train_loss": results.get("train_loss", 0.0),
                "train_accuracy": results.get("train_accuracy", 0.0),
                "train_f1": results.get("train_f1", 0.0),
                "train_mae": results.get("train_mae", 0.0),
                "val_loss": results.get("val_loss", 0.0),
                "val_accuracy": results.get("val_accuracy", 0.0),
                "val_f1": results.get("val_f1", 0.0),
                "val_mae": results.get("val_mae", 0.0),
                "training_time": training_time,
                "n_cpu": n_cpu,
                "ram": ram,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "communication_time": communication_time,
                "client_id": self.cid,
                "model_type": self.model_type,
                "data_distribution_type": data_distribution_type,
                "dataset": dataset,
                "compressed_parameters_hex": compressed_parameters_hex,
            }
            return [], len(self.trainloader.dataset), metrics
        else:
            metrics = {
                "train_loss": results.get("train_loss", 0.0),
                "train_accuracy": results.get("train_accuracy", 0.0),
                "train_f1": results.get("train_f1", 0.0),
                "train_mae": results.get("train_mae", 0.0),
                "val_loss": results.get("val_loss", 0.0),
                "val_accuracy": results.get("val_accuracy", 0.0),
                "val_f1": results.get("val_f1", 0.0),
                "val_mae": results.get("val_mae", 0.0),
                "training_time": training_time,
                "n_cpu": n_cpu,
                "ram": ram,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "communication_time": communication_time,
                "client_id": self.cid,
                "model_type": self.model_type,
                "data_distribution_type": data_distribution_type,
                "dataset": dataset,
            }
            return new_parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_weights_A(self.net, parameters)
        loss, accuracy = test_A(self.net, self.testloader)
        metrics = {
            "accuracy": accuracy,
            "client_id": self.cid,
            "model_type": self.model_type,
        }
        return loss, len(self.testloader.dataset), metrics

if __name__ == "__main__":
    details = load_client_details()
    client_id = os.getenv("CLIENT_ID")
    config = next((c for c in details if str(c.get("client_id")) == client_id), details[0])
    model_type = config.get("model")
    start_client(
        server_address=os.getenv("SERVER_ADDRESS", "server:8080"),
        client=FlowerClient(client_config=config, model_type=model_type).to_client()
    )
