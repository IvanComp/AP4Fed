import base64
import json
import os
import torch
import random
import time
import zlib
import pickle
import numpy as np
import psutil
import sys
import taskA
import torch
from pathlib import Path
import fcntl 
from datetime import datetime
from io import BytesIO
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log
from logging import INFO
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
        cfg_path = os.path.join(os.path.dirname(__file__), 'configuration', 'config.json')
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        details = cfg.get("client_details", [])
        try:
            details.sort(key=lambda x: int(x.get("client_id", 0)))
        except Exception:
            pass
        global_client_details = {str(d["client_id"]): d for d in details}
    return global_client_details

CLIENT_DETAILS = load_client_details()  
CLIENT_CONFIG_LIST = sorted(          
    CLIENT_DETAILS.values(), key=lambda c: int(c["client_id"])
)
COUNTER_PATH = Path(__file__).with_name(".client_idx")
if not COUNTER_PATH.exists():
    COUNTER_PATH.write_text("0")   

def get_next_config() -> dict:
    with COUNTER_PATH.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        idx = int(f.read().strip() or 0)
        config = CLIENT_CONFIG_LIST[idx % len(CLIENT_CONFIG_LIST)]
        f.seek(0)
        f.truncate()
        f.write(str((idx + 1) % len(CLIENT_CONFIG_LIST)))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return config

DISTRIBUTED_MODEL_REPAIR = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1 
SSIM = False

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

def set_cpu_affinity(process_pid: int, num_cpus: int) -> bool:
    import platform, os
    import psutil

    def safe_log(message):
        pass

    system = platform.system()
    try:
        process = psutil.Process(process_pid)
        total_cpus = os.cpu_count()
        cpus_to_use = min(num_cpus, total_cpus)

        if cpus_to_use <= 0:
            safe_log(f"Client (PID {process_pid}): Invalid or zero CPU count ({num_cpus}), skipping affinity.")
            return False

        target_cpus = list(range(cpus_to_use)) 

        if system == "Windows":
            process.cpu_affinity(target_cpus)
            safe_log(f"Client (PID {process_pid}): CPU affinity set to cores {target_cpus} on Windows.")
            return True
        elif system == "Linux":
            process.cpu_affinity(target_cpus)
            safe_log(f"Client (PID {process_pid}): CPU affinity set to cores {target_cpus} on Linux.")
            return True
        elif system == "Darwin":
            try:
                process.nice(10)  
                safe_log(f"Client (PID {process_pid}): Set nice=10 on macOS (CPU affinity not directly supported).")
            except psutil.AccessDenied:
                safe_log(f"Client (PID {process_pid}): Insufficient permissions to set nice value on macOS.")
            return True
        else:
            safe_log(f"Client (PID {process_pid}): OS '{system}' not supported for CPU affinity.")
            return False
    except psutil.NoSuchProcess:
        safe_log(f"Error: Process {process_pid} not found.")
        return False
    except psutil.AccessDenied:
        safe_log(f"Error: Permission denied when setting CPU affinity for process {process_pid}.")
        return False
    except Exception as e:
        safe_log(f"Unexpected error while setting CPU affinity for {process_pid}: {e}")
        return False

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

        self.cached_round_loaded = None
        if self.cached_round_loaded != GLOBAL_ROUND_COUNTER:
            self.trainloader, self.testloader = load_data_A(self.client_config, GLOBAL_ROUND_COUNTER)
            self.cached_round_loaded = GLOBAL_ROUND_COUNTER


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
                "jsd": get_jsd_A(self.trainloader) if HETEROGENEOUS_DATA_HANDLER else 0.0,
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
                "hdh_ms": hdh_ms if HETEROGENEOUS_DATA_HANDLER else 0.0,
                "communication_time": communication_time,
                "client_sent_ts": send_ready_ts,
                "train_end_ts": train_end_ts,
                "client_id": self.cid,
                "model_type": self.model_type,
                "data_distribution_type": self.data_distribution_type,
                "dataset": self.dataset,
                "jsd": get_jsd_A(self.trainloader) if HETEROGENEOUS_DATA_HANDLER else 0.0,
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

def client_fn(context: Context):
    config = get_next_config()
    cid_str = str(config["client_id"])
    return FlowerClient(client_config=config, model_type=config.get("model"))

app = ClientApp(client_fn=client_fn)