#!/usr/bin/env python3
from typing import List, Tuple, Dict, Optional
from flwr.common import (
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    Context,
    FitIns,
    EvaluateIns,
)
from flwr.server import (
    ServerConfig,
    ServerApp,
    ServerAppComponents,
    start_server
)
from io import BytesIO
from rich.panel import Panel
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO
import textwrap
import numpy as np
from taskA import Net as NetA, get_weights as get_weights_A, set_weights as set_weights_A, load_data as load_data_A
from rich.console import Console
import shutil
import time
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import json  
import zlib
import pickle
import docker
from APClient import ClientRegistry
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
folders_to_delete = ["performance", "model_weights"]

for folder in folders_to_delete:
    folder_path = os.path.join(current_dir, folder)
    if os.path.exists(folder_path):
        # svuota la cartella senza toccare il mount point
        for nome in os.listdir(folder_path):
            percorso = os.path.join(folder_path, nome)
            if os.path.isdir(percorso):
                shutil.rmtree(percorso, ignore_errors=True)
            else:
                try:
                    os.remove(percorso)
                except OSError:
                    pass
        
client_registry = ClientRegistry()

################### GLOBAL PARAMETERS
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MODEL_COVERSIONING, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MODEL_COVERSIONING = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False

# Non inizializziamo global_metrics con una chiave fissa, verrà creata dinamicamente
global_metrics = {}

matplotlib.use('Agg')
current_dir = os.path.abspath(os.path.dirname(__file__))

# Path to the 'configuration' directory
config_dir = os.path.join(current_dir, 'configuration') 
config_file = os.path.join(config_dir, 'config.json')

# Lettura dei parametri dal file di configurazione
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    num_rounds = int(config.get('rounds', 10))
    client_count = int(config.get('clients', 2))
    for pattern_name, pattern_info in config["patterns"].items():
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

    CLIENT_DETAILS = config.get("client_details", [])
    client_details_structure = []
    for client in CLIENT_DETAILS:
        client_details_structure.append({
            "client_id": client.get("client_id"),
            "cpu": client.get("cpu"),
            "ram": client.get("ram"),
            "cpu_percent": client.get("cpu_percent"),
            "ram_percent": client.get("ram_percent"),
            "dataset": client.get("dataset"),
            "data_distribution_type": client.get("data_distribution_type"),
            "model": client.get("model")
        })
    GLOBAL_CLIENT_DETAILS = client_details_structure

currentRnd = 0

performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Time of FL Round',
        '# of CPU', 'CPU Usage (%)', 'RAM Usage (%)',
        'Model Type', 'Data Distr. Type', 'Dataset',
        'Train Loss', 'Train Accuracy', 'Train F1', 'Train MAE',
        'Val Loss', 'Val Accuracy', 'Val F1', 'Val MAE'
    ])

# La funzione log_round_time ora riceve anche cpu_percent e ram_percent
def log_round_time(
     client_id, fl_round,
     training_time, communication_time, time_between_rounds,
     n_cpu, cpu_percent, ram_percent,
     client_model_type, data_distr, dataset_value,
     already_logged, srt1, srt2, agg_key
):
     # Assicuriamoci che i valori esistano
     if agg_key not in global_metrics:
         global_metrics[agg_key] = {
             "train_loss": [], "train_accuracy": [], "train_f1": [], "train_mae": [],
             "val_loss": [], "val_accuracy": [], "val_f1": [], "val_mae": []
         }
     # Prendo l'ultimo valore
     tm = global_metrics[agg_key]
     train_loss     = tm["train_loss"][-1]     if tm["train_loss"]     else None
     train_accuracy = tm["train_accuracy"][-1] if tm["train_accuracy"] else None
     train_f1       = tm["train_f1"][-1]       if tm["train_f1"]       else None
     train_mae      = tm["train_mae"][-1]      if tm["train_mae"]      else None
     val_loss       = tm["val_loss"][-1]       if tm["val_loss"]       else None
     val_accuracy   = tm["val_accuracy"][-1]   if tm["val_accuracy"]   else None
     val_f1         = tm["val_f1"][-1]         if tm["val_f1"]         else None
     val_mae        = tm["val_mae"][-1]        if tm["val_mae"]        else None

     # Per le righe non-last, vuoto i metrici e srt2
     if already_logged:
         srt2 = None

     # Scrivo i valori nella CSV nell’ordine giusto
     with open(csv_file, 'a', newline='') as file:
         writer = csv.writer(file)
         writer.writerow([
             client_id,
             fl_round + 1,
             f"{training_time:.2f}",        
             f"{communication_time:.2f}",   
             f"{time_between_rounds:.2f}",           
             n_cpu,   
             f"{cpu_percent:.0f}",                      
             f"{ram_percent:.0f}"  ,    
             client_model_type,
             data_distr,
             dataset_value,
             f"{train_loss:.2f}"     if train_loss    is not None else "",
             f"{train_accuracy:.4f}" if train_accuracy is not None else "",
             f"{train_f1:.4f}"       if train_f1       is not None else "",
             f"{train_mae:.4f}"      if train_mae      is not None else "",
             f"{val_loss:.2f}"       if val_loss       is not None else "",
             f"{val_accuracy:.4f}"   if val_accuracy   is not None else "",
             f"{val_f1:.4f}"         if val_f1         is not None else "",
             f"{val_mae:.4f}"        if val_mae        is not None else "",
         ])

def preprocess_csv():
    import pandas as pd
    import seaborn as sns

    df = pd.read_csv(csv_file)
    
    df["Client Number"] = (
        df["Client ID"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    df["Training Time"] = pd.to_numeric(df["Training Time"], errors="coerce")
    df["Total Time of FL Round"] = pd.to_numeric(
        df["Total Time of FL Round"], errors="coerce"
    )

    df["Total Time of FL Round"] = (
        df.groupby("FL Round")["Total Time of FL Round"]
        .transform(lambda x: [None] * (len(x) - 1) + [x.iloc[-1]])
    )

    df.sort_values(["FL Round", "Client Number"], inplace=True)
    cols_round = ["Total Time of FL Round"] + list(
        df.columns[df.columns.get_loc("Train Loss"):]
    )

    def fix_round_values(subdf):
        subdf = subdf.copy()
        last = subdf["Client Number"].max()
        for col in cols_round:
            vals = subdf[col].dropna()
            v = vals.iloc[-1] if not vals.empty else pd.NA
            subdf.loc[subdf["Client Number"] == last, col] = v
            subdf.loc[subdf["Client Number"] != last, col] = pd.NA
        return subdf

    df = df.groupby("FL Round", group_keys=False).apply(fix_round_values)
    df.drop(columns=["Client Number"], inplace=True)
    df.to_csv(csv_file, index=False)
    sns.set_theme(style="ticks")


def weighted_average_global(metrics, agg_model_type, srt1, srt2, time_between_rounds):
    if agg_model_type not in global_metrics:
        global_metrics[agg_model_type] = {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "train_mae": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_mae": []
        }
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    if total_examples == 0:
        return {
            "train_loss": float('inf'),
            "train_accuracy": 0.0,
            "train_f1": 0.0,
            "train_mae": 0.0,
            "val_loss": float('inf'),
            "val_accuracy": 0.0,
            "val_f1": 0.0,
            "val_mae": 0.0,
        }

    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    train_f1 = [num_examples * m["train_f1"] for num_examples, m in metrics]
    train_maes = [num_examples * m["train_mae"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_f1 = [num_examples * m["val_f1"] for num_examples, m in metrics]
    val_maes = [num_examples * m["val_mae"] for num_examples, m in metrics]

    avg_train_loss = sum(train_losses) / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_train_f1 = sum(train_f1) / total_examples
    avg_train_mae = sum(train_maes) / total_examples
    avg_val_loss = sum(val_losses) / total_examples
    avg_val_accuracy = sum(val_accuracies) / total_examples
    avg_val_f1 = sum(val_f1) / total_examples
    avg_val_mae = sum(val_maes) / total_examples

    global_metrics[agg_model_type]["train_loss"].append(avg_train_loss)
    global_metrics[agg_model_type]["train_accuracy"].append(avg_train_accuracy)
    global_metrics[agg_model_type]["train_f1"].append(avg_train_f1)
    global_metrics[agg_model_type]["train_mae"].append(avg_train_mae)
    global_metrics[agg_model_type]["val_loss"].append(avg_val_loss)
    global_metrics[agg_model_type]["val_accuracy"].append(avg_val_accuracy)
    global_metrics[agg_model_type]["val_f1"].append(avg_val_f1)
    global_metrics[agg_model_type]["val_mae"].append(avg_val_mae)

    client_data_list = []
    for num_examples, m in metrics:
        client_id = m.get("client_id")
        model_type = m.get("model_type", "N/A")
        data_distr = m.get("data_distribution_type", "N/A")
        dataset_value = m.get("dataset", "N/A")
        training_time = m.get("training_time")
        n_cpu = m.get("n_cpu")
        cpu_percent = m.get("cpu_percent")
        ram_percent = m.get("ram_percent")
        communication_time = m.get("communication_time")      
        if client_id:
            srt2 = time_between_rounds           
            total_time = training_time + communication_time
            client_data_list.append((
                client_id, training_time, communication_time, time_between_rounds,
                n_cpu, cpu_percent, ram_percent,
                model_type, data_distr, dataset_value, srt1, srt2
            ))

    num_clients = len(client_data_list)
    for idx, client_data in enumerate(client_data_list):
        (
            client_id, training_time, communication_time, time_between_rounds,
            n_cpu, cpu_percent, ram_percent,
            model_type, data_distr, dataset_value, srt1, srt2
        ) = client_data
        already_logged = (idx != num_clients - 1)
        log_round_time(
            client_id, currentRnd-1,
            training_time, communication_time, time_between_rounds,
            n_cpu, cpu_percent, ram_percent,
            model_type, data_distr, dataset_value,
            already_logged, srt1, srt2, agg_model_type
        )

    return {
        "train_loss": avg_train_loss,
        "train_accuracy": avg_train_accuracy,
        "train_f1": avg_train_f1,
        "train_mae": avg_train_mae,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
        "val_f1": avg_val_f1,
        "val_mae": avg_val_mae,
    }

parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
client_model_mapping = {}
previous_round_end_time = time.time() 

class MultiModelStrategy(Strategy):
    def __init__(self, initial_parameters_a: Parameters):
        self.parameters_a = initial_parameters_a
        banner = r"""
  ___  ______  ___ ______       _ 
 / _ \ | ___ \/   ||  ___|     | |
/ /_\ \| |_/ / /| || |_ ___  __| |
|  _  ||  __/ /_| ||  _/ _ \/ _` |
| | | || |  \___  || ||  __/ (_| |
\_| |_/\_|      |_/\_| \___|\__,_| v.1.5.0

"""
        log(INFO, "==========================================")
        for raw in banner.splitlines()[1:]:         
            line = raw.replace(" ", "\u00A0")        
            log(INFO, line)
        log(INFO, "==========================================")
        log(INFO, "Simulation Started!")
        log(INFO, "List of the Architectural Patterns enabled:")

        enabled_patterns = []
        for pattern_name, pattern_info in config["patterns"].items():
            if pattern_info["enabled"]:
                enabled_patterns.append((pattern_name, pattern_info))

        if not enabled_patterns:
            log(INFO, "No patterns are enabled.")
        else:
            for pattern_name, pattern_info in enabled_patterns:
                pattern_str = pattern_name.replace('_', ' ').title()
                log(INFO, f"{pattern_str} ✅")
                if pattern_info["params"]:
                    log(INFO, f" AP Parameters: {pattern_info['params']}")
                time.sleep(1)
        log(INFO, "==========================================")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        t0 = time.time()
        client_manager.wait_for(client_count) 
        clients = client_manager.sample(num_clients=client_count)
        fit_configurations = []

        if MESSAGE_COMPRESSOR:
            fake_tensors = []
            for tensor in self.parameters_a.tensors:
                buffer = BytesIO(tensor)
                loaded_array = np.load(buffer)
                reduced_shape = tuple(max(dim // 10, 1) for dim in loaded_array.shape)
                fake_array = np.zeros(reduced_shape, dtype=loaded_array.dtype)
                fake_serialized = BytesIO()
                np.save(fake_serialized, fake_array)
                fake_serialized.seek(0)
                fake_tensors.append(fake_serialized.read())
            fake_parameters = Parameters(tensors=fake_tensors, tensor_type=self.parameters_a.tensor_type)
            serialized_parameters = pickle.dumps(self.parameters_a)
            original_size = len(serialized_parameters)  
            compressed_parameters = zlib.compress(serialized_parameters)
            compressed_parameters_hex = compressed_parameters.hex()
            compressed_size = len(compressed_parameters)
            reduction_bytes = original_size - compressed_size
            reduction_percentage = (reduction_bytes / original_size) * 100
            log(INFO, f"Global Model Parameters compressed (from Server to Client) reduction of {reduction_bytes} bytes ({reduction_percentage:.2f}%)")

        for client in clients:
            if MESSAGE_COMPRESSOR:
                fit_ins = FitIns(fake_parameters, {"compressed_parameters_hex": compressed_parameters_hex})
            else:
                fit_ins = FitIns(self.parameters_a, {})
            fit_configurations.append((client, fit_ins))
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        global previous_round_end_time

        if previous_round_end_time is not None:
            time_between_rounds = time.time() - previous_round_end_time
            log(INFO, f"Results Aggregated in {time_between_rounds:.2f} seconds.")

        results_a = []
        training_times = []
        global currentRnd
        currentRnd += 1
        srt1 = 'N/A'
        srt2 = 'N/A'
        
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id")
            model_type = fit_res.metrics.get("model_type")
            training_time = fit_res.metrics.get("training_time")
            communication_time_client = fit_res.metrics.get("communication_time")
            recv_time = time.time()
            communication_time = recv_time - communication_time_client
            fit_res.metrics["communication_time"] = communication_time
            
            compressed_parameters_hex = fit_res.metrics.get("compressed_parameters_hex")
            client_model_mapping[client_id] = model_type

            if MESSAGE_COMPRESSOR:
                compressed_parameters = bytes.fromhex(compressed_parameters_hex)
                decompressed_parameters = pickle.loads(zlib.decompress(compressed_parameters))
                fit_res.parameters = ndarrays_to_parameters(decompressed_parameters)

            if training_time is not None:
                training_times.append(training_time)              
            results_a.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))

        previous_round_end_time = time.time()
        srt1 = max(training_times)
        agg_model_type = results_a[0][2].get("model_type", "Unknown")
        self.parameters_a = self.aggregate_parameters(results_a, agg_model_type, srt1, srt2, time_between_rounds)

        metrics_aggregated = {}
        if any(global_metrics[agg_model_type].values()):
            metrics_aggregated[agg_model_type] = {
                key: global_metrics[agg_model_type][key][-1] if global_metrics[agg_model_type][key] else None
                for key in global_metrics[agg_model_type]
            }

        aggregated_model = NetA()
        aggregated_params_list = parameters_to_ndarrays(self.parameters_a)
        set_weights_A(aggregated_model, aggregated_params_list)

        if MODEL_COVERSIONING:
            server_folder = os.path.join("model_weights", "server")
            os.makedirs(server_folder, exist_ok=True)
            server_file_path = os.path.join(server_folder, f"MW_round{currentRnd}.pt")
            torch.save(aggregated_model.state_dict(), server_file_path)
            log(INFO, f"Aggregated model weights saved to {server_file_path}")
        
        preprocess_csv()

        # crea una copia con il numero del round
        round_csv = os.path.join(
            performance_dir,
            f"FLwithAP_performance_metrics_round{currentRnd}.csv"
        )
        shutil.copy(csv_file, round_csv)

        return (self.parameters_a), metrics_aggregated

    def aggregate_parameters(self, results, agg_model_type, srt1, srt2, time_between_rounds):
        total_examples = sum([num_examples for _, num_examples, _ in results])
        new_weights = None

        metrics = []
        for client_params, num_examples, client_metrics in results:
            client_weights = parameters_to_ndarrays(client_params)
            weight = num_examples / total_examples
            if new_weights is None:
                new_weights = [w * weight for w in client_weights]
            else:
                new_weights = [nw + w * weight for nw, w in zip(new_weights, client_weights)]
            metrics.append((num_examples, client_metrics))

        weighted_average_global(metrics, agg_model_type, srt1, srt2, time_between_rounds)
        return ndarrays_to_parameters(new_weights)
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        #log(INFO, f"Evaluating Performance Metrics...")
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        return None

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    
        if server_round == 0:
            return None
        
        #log(INFO, f"[ROUND {server_round}] Evaluating Performance Metrics...")

        # ADAPTATION CODE

        #log(INFO, f"[ROUND {server_round+1}] Adapting new settings...")

        return None

if __name__ == "__main__":

    strategy = MultiModelStrategy(
        initial_parameters_a=parametersA,  
    )

    start_server(
        server_address="[::]:8080",  
        config=ServerConfig(num_rounds=num_rounds),  
        strategy=strategy, 
    )
