from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from typing import Dict
import json
from datetime import datetime
import os
import hashlib
import torch
from io import BytesIO
import zlib
import pickle
import numpy as np
from flwr.common.logger import log
from logging import INFO
from APClient import ClientRegistry
from taskA import (
    DEVICE as DEVICE_A,
    Net as NetA,
    get_weights as get_weights_A,
    load_data as load_data_A,
    set_weights as set_weights_A,
    train as train_A,
    test as test_A
)
from taskB import (
    DEVICE as DEVICE_B,
    Net as NetB,
    get_weights as get_weights_B,
    load_data as load_data_B,
    set_weights as set_weights_B,
    train as train_B,
    test as test_B
)
import threading
import ray
import psutil
import resource

@ray.remote
class ConfigServer:
    def __init__(self, config_list):
        self.config_list = config_list
        self.counter = 0
        self.assignments = {}
    def get_config_for(self, client_id: str):
        if client_id in self.assignments:
            return self.assignments[client_id]
        else:
            if self.counter < len(self.config_list):
                config = self.config_list[self.counter]
                self.assignments[client_id] = config
                self.counter += 1
                return config

global_client_details = None
def load_client_details():
    global global_client_details
    if global_client_details is None:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        config_dir = os.path.join(current_dir, '..', 'configuration')
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

CLIENT_REGISTRY = ClientRegistry()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    def __init__(self, client_config: dict, model_type: str):
        self.client_config = client_config
        self.cid = client_config.get("client_id", "unknown")
        self.n_cpu = client_config.get("cpu")
        self.ram = client_config.get("ram")
        self.dataset = client_config.get("dataset")
        self.data_distribution_type = client_config.get("data_distribution_type")
        self.model_type = model_type
        #print(f"[DEBUG] Creating FlowerClient: id={self.cid}, cpu (config)={self.n_cpu}, ram (config)={self.ram}, dataset={self.dataset}, distribution={self.data_distribution_type}")
        CLIENT_REGISTRY.register_client(self.cid, model_type)
        self.net = NetA().to(DEVICE_A)
        self.trainloader, self.testloader = load_data_A()
        self.device = DEVICE_A
    def fit(self, parameters, config):
        compressed_parameters_hex = config.get("compressed_parameters_hex")
        global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
        CLIENT_SELECTOR = False
        CLIENT_CLUSTER = False
        MESSAGE_COMPRESSOR = False
        MULTI_TASK_MODEL_TRAINER = False
        HETEROGENEOUS_DATA_HANDLER = False
        current_dir = os.path.abspath(os.path.dirname(__file__))
        config_dir = os.path.join(current_dir, '..', 'configuration')
        config_file = os.path.join(config_dir, 'config.json')
        numpy_arrays = None
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                configJSON = json.load(f)
            for pattern_name, pattern_info in configJSON["patterns"].items():
                if pattern_info["enabled"]:
                    if pattern_name == "client_selector":
                        CLIENT_SELECTOR = True
                    elif pattern_name == "client_cluster":
                        CLIENT_CLUSTER = True
                    elif pattern_name == "message_compressor":
                        MESSAGE_COMPRESSOR = True
                    elif pattern_name == "multi-task_model_trainer":
                        MULTI_TASK_MODEL_TRAINER = True
                    elif pattern_name == "heterogeneous_data_handler":
                        HETEROGENEOUS_DATA_HANDLER = True

        #Extracting client specifications 
        n_cpu = self.n_cpu
        ram = self.ram
        dataset = self.dataset
        data_distribution_type = self.data_distribution_type

        if CLIENT_SELECTOR:
            selector_params = configJSON["patterns"]["client_selector"]["params"]
            selection_strategy = selector_params.get("selection_strategy", "")
            selection_criteria = selector_params.get("selection_criteria", "")
            selection_value = selector_params.get("selection_value", "")
            if selection_strategy == "Resource-Based":
                if selection_criteria == "CPU":
                    if n_cpu < selection_value:
                        log(INFO, f"Client {self.cid} has insufficient CPU ({n_cpu} / {selection_value}). Will not participate in this FL round.")
                        log(INFO, "Preparing empty response")
                        return parameters, 0, {}
                    else:
                        log(INFO, f"Client {self.cid} participates in the FL round. (CPU: {n_cpu})")
                elif selection_criteria == "RAM":
                    if ram < selection_value:
                        log(INFO, f"Client {self.cid} has insufficient RAM ({ram} GB / {selection_value} GB). Will not participate in this FL round.")
                        log(INFO, "Preparing empty response")
                        return parameters, 0, {}
                    else:
                        log(INFO, f"Client {self.cid} participates in the FL round. (RAM: {ram} GB)")
        if CLIENT_CLUSTER:
            selector_params = configJSON["patterns"]["client_cluster"]["params"]
            clustering_strategy = selector_params.get("clustering_strategy", "")
            clustering_criteria = selector_params.get("clustering_criteria", "")
            selection_value = selector_params.get("selection_value", "")
            if clustering_strategy == "Resource-Based":
                if clustering_criteria == "CPU":
                    if n_cpu < selection_value:
                        log(INFO, f"Client {self.cid} assigned to Cluster A {self.model_type}")
                    else:
                        log(INFO, f"Client {self.cid} assigned to Cluster B {self.model_type}")
                elif clustering_criteria == "RAM":
                    if ram < selection_value:
                        log(INFO, f"Client {self.cid} assigned to Cluster A {self.model_type}")
                    else:
                        log(INFO, f"Client {self.cid} assigned to Cluster B {self.model_type}")
            elif clustering_strategy == "Data-Based":
                if clustering_criteria == "IID":
                    log(INFO, f"Client {self.cid} assigned to IID Cluster {self.model_type}")
                elif clustering_criteria == "non-IID":
                    log(INFO, f"Client {self.cid} assigned to non-IID Cluster {self.model_type}")
        if MESSAGE_COMPRESSOR:
            compressed_parameters = bytes.fromhex(compressed_parameters_hex)
            decompressed_parameters = pickle.loads(zlib.decompress(compressed_parameters))
            numpy_arrays = [np.load(BytesIO(tensor)) for tensor in decompressed_parameters.tensors]
            numpy_arrays = [arr.astype(np.float32) for arr in numpy_arrays]
            parameters = numpy_arrays
        else:
            parameters = parameters
        set_weights_A(self.net, parameters)
        results, training_time, start_comm_time = train_A(self.net, self.trainloader, self.testloader, epochs=1, device=self.device)
        new_parameters = get_weights_A(self.net)
        compressed_parameters_hex = None
        if MESSAGE_COMPRESSOR:
            serialized_parameters = pickle.dumps(new_parameters)
            original_size = len(serialized_parameters)
            compressed_parameters = zlib.compress(serialized_parameters)
            compressed_size = len(compressed_parameters)
            compressed_parameters_hex = compressed_parameters.hex()
            reduction_bytes = original_size - compressed_size
            reduction_percentage = (reduction_bytes / original_size) * 100
            log(INFO, f"Local parameters compressed (Client -> Server): reduced by {reduction_bytes} bytes ({reduction_percentage:.2f}%)")
            metrics = {
                "train_loss": results["train_loss"],
                "train_accuracy": results["train_accuracy"],
                "train_f1": results["train_f1"],
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_f1": results["val_f1"],
                "training_time": training_time,
                "cpu_usage": n_cpu,
                "ram": ram,
                "client_id": self.cid,
                "model_type": self.model_type,
                "start_comm_time": start_comm_time,
                "compressed_parameters_hex": compressed_parameters_hex,
            }
        else:
            metrics = {
                "train_loss": results["train_loss"],
                "train_accuracy": results["train_accuracy"],
                "train_f1": results["train_f1"],
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_f1": results["val_f1"],
                "training_time": training_time,
                "cpu_usage": n_cpu,
                "ram": ram,
                "client_id": self.cid,
                "model_type": self.model_type,
                "start_comm_time": start_comm_time,
            }
        if MESSAGE_COMPRESSOR:
            return [], len(self.trainloader.dataset), metrics
        else:
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

def query_cpu():
    try:
        with open("/sys/fs/cgroup/cpu.max", "rt") as f:
            cfs_quota_us, cfs_period_us = [int(v) for v in f.read().strip().split()]
            cpu_quota = cfs_quota_us // cfs_period_us
    except FileNotFoundError:
        cpu_quota = os.cpu_count()
    return cpu_quota

def query_ram():
    try:
        with open("/sys/fs/cgroup/memory.max", "rt") as f:
            memory_limit = int(f.read().strip())
            if memory_limit == -1:
                total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            else:
                total_memory = memory_limit
    except FileNotFoundError:
        total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    return total_memory // (1024 * 1024 * 1024)

def client_fn(context: Context):
    client_identifier = context.node_id
    try:
        config_server = ray.get_actor("config_server")
    except Exception:
        details = load_client_details()
        try:
            config_server = ConfigServer.options(name="config_server", lifetime="detached").remote(details)
        except Exception:
            config_server = ray.get_actor("config_server")
    config = ray.get(config_server.get_config_for.remote(client_identifier))
    model_type = "taskA"
    return FlowerClient(client_config=config, model_type=model_type).to_client()

app = ClientApp(client_fn=client_fn)
