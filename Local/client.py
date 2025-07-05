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
import sys
import taskA
import torch
from pathlib import Path
import fcntl 
import logging
logging.getLogger("onnx2keras").setLevel(logging.ERROR)
import onnx
from onnx2keras import onnx_to_keras
from datetime import datetime
from io import BytesIO
from flwr.client import ClientApp, NumPyClient
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
        # ordino per client_id
        try:
            details.sort(key=lambda x: int(x.get("client_id", 0)))
        except Exception:
            pass
        # mappa id → dict
        global_client_details = {str(d["client_id"]): d for d in details}
    return global_client_details

CLIENT_DETAILS = load_client_details()  # resta
CLIENT_CONFIG_LIST = sorted(           # resta
    CLIENT_DETAILS.values(), key=lambda c: int(c["client_id"])
)
COUNTER_PATH = Path(__file__).with_name(".client_idx")
if not COUNTER_PATH.exists():
    COUNTER_PATH.write_text("0")   

def get_next_config() -> dict:
    """Legge/aggiorna in modo atomico il contatore nel file e
    restituisce la configurazione da assegnare a questo processo."""
    with COUNTER_PATH.open("r+") as f:
        # lock esclusivo
        fcntl.flock(f, fcntl.LOCK_EX)
        idx = int(f.read().strip() or 0)
        config = CLIENT_CONFIG_LIST[idx % len(CLIENT_CONFIG_LIST)]
        f.seek(0)
        f.truncate()
        f.write(str((idx + 1) % len(CLIENT_CONFIG_LIST)))
        f.flush()
        # rilascia il lock
        fcntl.flock(f, fcntl.LOCK_UN)
    return config

CLIENT_REGISTRY = ClientRegistry()
DISTRIBUTED_MODEL_REPAIR = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1 
SSIM = False

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
        self.cid = client_config.get("client_id", "unknown")
        self.n_cpu = client_config.get("cpu")
        self.ram = client_config.get("ram")
        self.dataset = client_config.get("dataset")
        self.data_distribution_type = client_config.get("data_distribution_type")
        self.model = client_config.get("model")
        self.model_type = model_type

        current_os = platform.system()

        def get_parameters(self, config):
            return get_parameters(self.net)

        def safe_log(message):
            #log(INFO, message)
            pass

        if self.n_cpu is not None:
            try:
                num_cpus_int = int(self.n_cpu)
                if num_cpus_int > 0:
                    set_cpu_affinity(os.getpid(), num_cpus_int)
                    if current_os == "Linux" or current_os == "Windows":
                        try:
                            current_affinity = psutil.Process(os.getpid()).cpu_affinity()
                            safe_log(f"Client {self.cid}: Affinità CPU attuale: {current_affinity}")
                        except Exception as e:
                            safe_log(f"Client {self.cid}: Impossibile ottenere affinità CPU attuale su {current_os}: {e}")
                    elif current_os == "Darwin":
                        safe_log(f"Client {self.cid}: Lettura affinità CPU non supportata su macOS.")
                else:
                    safe_log(f"Client {self.cid}: Valore CPU non positivo ({self.n_cpu}), non imposto affinità.")
            except (ValueError, TypeError):
                safe_log(f"Client {self.cid}: Valore CPU non valido ({self.n_cpu}), non imposto affinità.")
        else:
            safe_log(f"Client {self.cid}: Parametro 'cpu' non specificato nella configurazione, non imposto affinità.")
            
        CLIENT_REGISTRY.register_client(self.cid, model_type)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = NetA().to(device)
        self.trainloader, self.testloader = load_data_A(self.client_config)
        self.DEVICE = device

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
        config_dir = os.path.join(current_dir, 'configuration')
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
                    elif pattern_name == "model_co-versioning_registry":
                        MODEL_COVERSIONING = True
                    elif pattern_name == "multi-task_model_trainer":
                        MULTI_TASK_MODEL_TRAINER = True
                    elif pattern_name == "heterogeneous_data_handler":
                        HETEROGENEOUS_DATA_HANDLER = True

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
                if selection_criteria == "CPU" and n_cpu < selection_value:
                    log(INFO, f"Client {self.cid} has insufficient CPU ({n_cpu}). Will not participate.")
                    return parameters, 0, {}
                if selection_criteria == "RAM" and ram < selection_value:
                    log(INFO, f"Client {self.cid} has insufficient RAM ({ram}). Will not participate.")
                    return parameters, 0, {}
            log(INFO, f"Client {self.cid} participates in this round.")

            if selection_strategy == "SSIM-Based":
                SSIM = True
                log(INFO, f"Entering SSIM.")

        if CLIENT_CLUSTER:
            selector_params = configJSON["patterns"]["client_cluster"]["params"]
            clustering_strategy = selector_params.get("clustering_strategy", "")
            clustering_criteria = selector_params.get("clustering_criteria", "")
            selection_value = selector_params.get("selection_value", "")
            if clustering_strategy == "Resource-Based":
                if clustering_criteria == "CPU":
                    grp = "A" if n_cpu < selection_value else "B"
                else:
                    grp = "A" if ram < selection_value else "B"
                log(INFO, f"Client {self.cid} assigned to Cluster {grp} {self.model_type}")
            elif clustering_strategy == "Data-Based":
                if clustering_criteria == "IID":
                    log(INFO, f"Client {self.cid} assigned to IID Cluster {self.model_type}")
                else:
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

        if DISTRIBUTED_MODEL_REPAIR and GLOBAL_ROUND_COUNTER > 1:
            log(INFO, f"Client {self.cid}: avvio distributed model repair")
            self.net.eval()
            X_batch, _ = next(iter(self.trainloader))
            input_shape = (1,) + tuple(X_batch.shape[1:])  

            onnx_path = "temp_model.onnx"
            dummy = torch.randn(input_shape).to(DEVICE)
            torch.onnx.export(
                self.net, dummy, onnx_path,
                input_names=["input"], output_names=["output"],
                opset_version=13
            )

            model_onnx = onnx.load(onnx_path)

            for node in model_onnx.graph.node:
                node.name = node.name.replace('/', '_')
                for i in range(len(node.input)):
                    node.input[i] = node.input[i].replace('/', '_')
                for i in range(len(node.output)):
                    node.output[i] = node.output[i].replace('/', '_')

            for init in model_onnx.graph.initializer:
                init.name = init.name.replace('/', '_')

            for v in model_onnx.graph.input:
                v.name = v.name.replace('/', '_')
            for v in model_onnx.graph.output:
                v.name = v.name.replace('/', '_')

            onnx.save(model_onnx, onnx_path)

            import keras.layers as _layers

            def _patch_init(layer_cls):
                orig = layer_cls.__init__
                def wrapped(self, *args, **kwargs):
                    w = kwargs.pop("weights", None)
                    orig(self, *args, **kwargs)
                    if w is not None and hasattr(self, "set_weights") and len(self.weights) > 0:
                        self.set_weights(w)
                layer_cls.__init__ = wrapped

            _patch_init(_layers.Conv2D)
            _patch_init(_layers.Dense)

            keras_model = onnx_to_keras(model_onnx, ["input"])
            keras_path = "model_local.h5"
            keras_model.save(keras_path)

            from arachne.run_localise import compute_FI_and_GL

            try:
                X_batch, y_batch = next(iter(self.trainloader))
                X_np = X_batch.cpu().numpy()
                y_np = y_batch.cpu().numpy()
            except StopIteration:
                log(INFO, f"Client {self.cid}: trainloader vuoto, skip repair")
            else:
                total_cands = compute_FI_and_GL(
                    X_np, y_np,
                    indices_to_target=list(range(len(X_np))),
                    target_weights={},
                    is_multi_label=False,
                    path_to_keras_model=keras_path
                )
                sd = self.net.state_dict()
                for layer_name, idxs, vals in total_cands:
                    w = sd[layer_name].cpu().numpy()
                    for idx, val in zip(idxs, vals):
                        w.flat[idx] = val
                    sd[layer_name].copy_(torch.from_numpy(w).to(DEVICE))
                self.net.load_state_dict(sd)
            self.net.train()

        results, training_time = train_A(
            self.net, self.trainloader, self.testloader,
            epochs=5, DEVICE=self.DEVICE
        )
        train_end_ts = taskA.TRAIN_COMPLETED_TS or time.time()
        new_parameters = get_weights_A(self.net)
        compressed_parameters_hex = None
        send_ready_ts = time.time()
        communication_time = send_ready_ts - train_end_ts
        wall_end = time.time()
        cpu_end = proc.cpu_times().user + proc.cpu_times().system
        cpu_percent = (cpu_end - cpu_start) / (wall_end - wall_start) * 100
        ram_percent = proc.memory_percent()
        
        round_number = GLOBAL_ROUND_COUNTER
        GLOBAL_ROUND_COUNTER += 1

        if MODEL_COVERSIONING:          
            base_path = os.path.dirname(os.path.abspath(__file__)) 
            client_folder = os.path.join(base_path, "model_weights", "clients", str(self.cid))
            os.makedirs(client_folder, exist_ok=True)
            client_file_path = os.path.join(client_folder, f"MW_round{round_number}.pt")
            torch.save(self.net.state_dict(), client_file_path)
            log(INFO, f"Client {self.cid} model weights saved to {client_file_path}")

            #if SSIM:
             #   log(INFO, f"Entering SSIM after model.")
                
        if MESSAGE_COMPRESSOR:
            serialized_parameters = pickle.dumps(new_parameters)
            original_size = len(serialized_parameters)
            compressed_parameters = zlib.compress(serialized_parameters)
            compressed_size = len(compressed_parameters)
            compressed_parameters_hex = compressed_parameters.hex()
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
                "train_loss": results["train_loss"],
                "train_accuracy": results["train_accuracy"],
                "train_f1": results["train_f1"],
                "train_mae": results.get("train_mae", 0.0),
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_f1": results["val_f1"],
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

def client_fn(context: Context):
    config = get_next_config()
    cid_str = str(config["client_id"])
    #log(INFO, f"Assigned config to client: {config}, "
    #    f"model_type: {config.get('model')}, client_id: {cid_str}")
    return FlowerClient(client_config=config, model_type=config.get("model"))

app = ClientApp(client_fn=client_fn)