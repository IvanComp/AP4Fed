import base64
import json
import os
import pickle
import platform
import random
import sys
import time
import zlib
import numpy as np
import psutil
import torch
import taskA
from io import BytesIO
from flwr.client import NumPyClient, start_client
from flwr.common.logger import log
from logging import INFO
from APClient import ClientRegistry
from taskA import (
    Net as NetA,
    get_weights as get_weights_A,
    load_data as load_data_A,
    set_weights as set_weights_A,
    train as train_A,
    test as test_A,
    get_jsd as get_jsd_A,
    rebalance_trainloader_with_gan as rebalance_trainloader_with_gan_A
)

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "arachne"
        )
    )
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
DISTRIBUTED_MODEL_REPAIR = False
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


def get_ram_percent_cgroup():
    paths = [
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
         "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for used_path, limit_path in paths:
        if os.path.exists(used_path) and os.path.exists(limit_path):
            try:
                with open(used_path) as f:
                    used = int(f.read())
                with open(limit_path) as f:
                    limit = int(f.read())
                if limit > 0:
                    return used / limit * 100
            except Exception:
                break

    return psutil.Process(os.getpid()).memory_percent()


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
        self.cid = f"Client {client_config.get('client_id')}"
        self.n_cpu = client_config.get("cpu")
        self.ram = client_config.get("ram")
        self.dataset = client_config.get("dataset")
        self.data_distribution_type = client_config.get("data_distribution_type")
        self.data_persistence_type = client_config.get("data_persistence_type", "Same Data")
        self.model = client_config.get("model")
        self.model_type = model_type
        self.did_hdh = False
        self.trainloader, self.testloader = None, None
        self.delay_enabled = (client_config.get("delay_combobox") == "Yes")
        self.delay_injection = 50

        if self.n_cpu is not None:
            try:
                num_cpus_int = int(self.n_cpu)
                if num_cpus_int > 0:
                    set_cpu_affinity(os.getpid(), num_cpus_int)
            except Exception:
                pass

        CLIENT_REGISTRY.register_client(self.cid, model_type)
        self.net = NetA().to(DEVICE)
        self.DEVICE = DEVICE

    def fit(self, parameters, config):
        global GLOBAL_ROUND_COUNTER
        hdh_ms = 0.0
        proc = psutil.Process(os.getpid())
        cpu_start = proc.cpu_times().user + proc.cpu_times().system
        wall_start = time.time()
        compressed_parameters_hex = config.get("compressed_parameters_hex")
        global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MODEL_COVERSIONING, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
        CLIENT_SELECTOR = CLIENT_CLUSTER = MESSAGE_COMPRESSOR = MODEL_COVERSIONING = MULTI_TASK_MODEL_TRAINER = HETEROGENEOUS_DATA_HANDLER = False
        current_dir = os.path.abspath(os.path.dirname(__file__))
        config_dir = os.path.join(current_dir, 'configuration')
        config_file = os.path.join(config_dir, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                configJSON = json.load(f)
            ADAPTATION_ENABLED = configJSON.get("adaptation", False)
            for name, info in configJSON.get("patterns", {}).items():
                if info.get("enabled"):
                    if name == "client_selector":
                        CLIENT_SELECTOR = True
                    elif name == "client_cluster":
                        CLIENT_CLUSTER = True
                    elif name == "message_compressor":
                        MESSAGE_COMPRESSOR = True
                    elif name == "model_co-versioning_registry":
                        MODEL_COVERSIONING = True
                    elif name == "multi-task_model_trainer":
                        MULTI_TASK_MODEL_TRAINER = True
                    elif name == "heterogeneous_data_handler":
                        if not ADAPTATION_ENABLED or self.cid in info.get("params", {}).get("enabled_clients", []):
                            HETEROGENEOUS_DATA_HANDLER = True

        never_loaded_data = self.trainloader is None and self.testloader is None
        needs_to_reload_data = self.data_persistence_type != "Same Data" and not never_loaded_data
        if never_loaded_data or needs_to_reload_data:
            self.trainloader, self.testloader = load_data_A(self.client_config, GLOBAL_ROUND_COUNTER)

        if HETEROGENEOUS_DATA_HANDLER and (ADAPTATION_ENABLED or not self.did_hdh):
            self.trainloader, hdh_ms = rebalance_trainloader_with_gan_A(self.trainloader)
            self.did_hdh = True

        if CLIENT_SELECTOR:
            selector_params = configJSON["patterns"]["client_selector"]["params"]
            selection_strategy = selector_params.get("selection_strategy", "")
            selection_criteria = selector_params.get("selection_criteria", "")
            selection_value = selector_params.get("selection_value", "")
            if selection_strategy == "Resource-Based":
                if selection_criteria == "CPU" and self.n_cpu < selection_value:
                    log(INFO,
                        f"Client {self.cid} has insufficient CPU ({self.n_cpu}). Will not participate in the next FL round.")
                    return parameters, 0, {}
                if selection_criteria == "RAM" and self.ram < selection_value:
                    log(INFO,
                        f"Client {self.cid} has insufficient RAM ({self.ram}). Will not participate in the next FL round.")
                    return parameters, 0, {}
            log(INFO, f"Client {self.cid} participates in this round. (CPU: {self.n_cpu}, RAM: {self.ram})")

        if CLIENT_CLUSTER:
            selector_params = configJSON["patterns"]["client_cluster"]["params"]
            clustering_strategy = selector_params.get("clustering_strategy", "")
            clustering_criteria = selector_params.get("clustering_criteria", "")
            selection_value = selector_params.get("selection_value", "")
            if clustering_strategy == "Resource-Based":
                if clustering_criteria == "CPU":
                    grp = "A" if self.n_cpu < selection_value else "B"
                else:
                    grp = "A" if self.ram < selection_value else "B"
                log(INFO, f"Client {self.cid} assigned to Cluster {grp} {self.model_type}")
            elif clustering_strategy == "Data-Based":
                if clustering_criteria == "IID":
                    log(INFO, f"Client {self.cid} assigned to IID Cluster {self.model_type}")
                else:
                    log(INFO, f"Client {self.cid} assigned to non-IID Cluster {self.model_type}")

        if MESSAGE_COMPRESSOR:
            payload_b64 = config.get("compressed_parameters_b64")
            compressed_parameters = base64.b64decode(payload_b64)
            decompressed_parameters = pickle.loads(zlib.decompress(compressed_parameters))
            numpy_arrays = [np.load(BytesIO(tensor)) for tensor in decompressed_parameters.tensors]
            numpy_arrays = [arr.astype(np.float32) for arr in numpy_arrays]
            parameters = numpy_arrays
        else:
            parameters = parameters

        set_weights_A(self.net, parameters)
        results, training_time = train_A(self.net, self.trainloader, self.testloader, epochs=1, DEVICE=self.DEVICE)
        new_parameters = get_weights_A(self.net)
        compressed_parameters_hex = None

        train_end_ts = taskA.TRAIN_COMPLETED_TS or time.time()
        if self.delay_enabled:
            random_delay = random.randint(0, self.delay_injection)
            log(INFO, f"client {self.cid} injecting delay of {random_delay} seconds")
            time.sleep(random_delay)
        send_ready_ts = time.time()
        communication_time = send_ready_ts - train_end_ts

        wall_end = time.time()
        cpu_end = proc.cpu_times().user + proc.cpu_times().system
        duration = wall_end - wall_start
        cpu_percent = ((cpu_end - cpu_start) / duration * 100) if duration > 0 else 0.0
        ram_percent = get_ram_percent_cgroup()

        round_number = GLOBAL_ROUND_COUNTER
        GLOBAL_ROUND_COUNTER += 1

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
            compressed_size = len(compressed_parameters)
            compressed_parameters_b64 = base64.b64encode(compressed_parameters).decode("ascii")
            reduction_bytes = original_size - compressed_size
            reduction_percentage = (reduction_bytes / original_size) * 100
            log(INFO, f"Local parameters compressed: reduced {reduction_bytes} bytes ({reduction_percentage:.2f}%)")
            metrics = {
                "train_loss": results["train_loss"],
                "train_accuracy": results["train_accuracy"],
                "train_f1": results["train_f1"],
                "train_mae": results.get("train_mae", 0.0),
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_f1": results["val_f1"],
                "val_mae": results.get("val_mae", 0.0),
                "training_time": training_time,
                "n_cpu": self.n_cpu,
                "ram": self.ram,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "hdh_ms": hdh_ms if HETEROGENEOUS_DATA_HANDLER else 0.0,
                "communication_time": communication_time,
                "client_sent_ts": send_ready_ts,
                "train_end_ts": train_end_ts,
                "client_id": self.cid,
                "model_type": self.model_type,
                "data_distribution_type": self.data_distribution_type,
                "dataset": self.dataset,
                "compressed_parameters_b64": compressed_parameters_b64,
            }
            return [], len(self.trainloader.dataset), metrics
        else:
            metrics = {
                "train_loss": results["train_loss"],
                "train_accuracy": results["train_accuracy"],
                "train_f1": results["train_f1"],
                "train_mae": results.get("train_mae", 0.0),
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_f1": results["val_f1"],
                "val_mae": results.get("val_mae", 0.0),
                "training_time": training_time,
                "n_cpu": self.n_cpu,
                "ram": self.ram,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "hdh_ms": hdh_ms if hdh_ms is not None else 0.0,
                "communication_time": communication_time,
                "client_sent_ts": send_ready_ts,
                "train_end_ts": train_end_ts,
                "client_id": self.cid,
                "model_type": self.model_type,
                "data_distribution_type": self.data_distribution_type,
                "dataset": self.dataset,
                "jsd": get_jsd_A(self.trainloader)
            }
            return new_parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_weights_A(self.net, parameters)
        loss, accuracy = test_A(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {
            "accuracy": accuracy,
            "client_id": self.cid,
            "model_type": self.model_type,
        }

if __name__ == "__main__":
    details = load_client_details()
    cid_env = os.getenv("CLIENT_ID")
    config = next((c for c in details if str(c.get("client_id")) == cid_env), details[0])
    model_type = config.get("model")
    start_client(
        server_address=os.getenv("SERVER_ADDRESS", "server:8080"),
        client=FlowerClient(client_config=config, model_type=model_type).to_client()
    )