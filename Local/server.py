#!/usr/bin/env python3
from typing import List, Tuple, Dict, Optional
import contextlib
import os
import builtins
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
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
from flwr.server.strategy import Strategy, FedAvg
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
        shutil.rmtree(folder_path)
        
client_registry = ClientRegistry()

################### GLOBAL PARAMETERS
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MODEL_COVERSIONING, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MODEL_COVERSIONING = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False

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

selector_params = config["patterns"]["client_selector"]["params"]
selection_strategy = selector_params.get("selection_strategy")      
selection_criteria = selector_params.get("selection_criteria")

MODEL_NAME = GLOBAL_CLIENT_DETAILS[0]["model"]
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

    df["Client ID"] = (
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
    mapping = {cid: cid for cid in sorted(df["Client ID"].unique())}
    df["Client Number"] = df["Client ID"].map(mapping)
    df["Client ID"] = df["Client ID"].map(lambda x: f"Client {x}")
    df.sort_values(["FL Round", "Client Number"], inplace=True)
    cols_round = ["Total Time of FL Round"] + list(
        df.columns[df.columns.get_loc("Train Loss") :]
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

    examples = [n for n, _ in metrics]
    total_examples = sum(examples)
    if total_examples == 0:
        return {
            "train_loss": float("inf"),
            "train_accuracy": 0.0,
            "train_f1": 0.0,
            "train_mae": 0.0,
            "val_loss": float("inf"),
            "val_accuracy": 0.0,
            "val_f1": 0.0,
            "val_mae": 0.0,
        }

    train_losses     = [n * m.get("train_loss",    0) for n, m in metrics]
    train_accuracies = [n * m.get("train_accuracy",0) for n, m in metrics]
    train_f1s        = [n * m.get("train_f1",      0) for n, m in metrics]
    train_maes       = [n * m.get("train_mae",     0) for n, m in metrics]
    val_losses       = [n * m.get("val_loss",      0) for n, m in metrics]
    val_accuracies   = [n * m.get("val_accuracy",  0) for n, m in metrics]
    val_f1s          = [n * m.get("val_f1",        0) for n, m in metrics]
    val_maes         = [n * m.get("val_mae",       0) for n, m in metrics]

    avg_train_loss     = sum(train_losses)     / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_train_f1       = sum(train_f1s)        / total_examples
    avg_train_mae      = sum(train_maes)       / total_examples
    avg_val_loss       = sum(val_losses)       / total_examples
    avg_val_accuracy   = sum(val_accuracies)   / total_examples
    avg_val_f1         = sum(val_f1s)          / total_examples
    avg_val_mae        = sum(val_maes)         / total_examples

    gm = global_metrics[agg_model_type]
    gm["train_loss"].append(avg_train_loss)
    gm["train_accuracy"].append(avg_train_accuracy)
    gm["train_f1"].append(avg_train_f1)
    gm["train_mae"].append(avg_train_mae)
    gm["val_loss"].append(avg_val_loss)
    gm["val_accuracy"].append(avg_val_accuracy)
    gm["val_f1"].append(avg_val_f1)
    gm["val_mae"].append(avg_val_mae)

    client_data_list = []
    for n, m in metrics:
        if n == 0:
            continue

        client_id          = m.get("client_id")
        model_type         = m.get("model_type", "N/A")
        data_distr         = m.get("data_distribution_type", "N/A")
        dataset_value      = m.get("dataset", "N/A")
        training_time      = m.get("training_time")      or 0.0
        communication_time = m.get("communication_time") or 0.0
        n_cpu              = m.get("n_cpu")              or 0
        cpu_percent        = m.get("cpu_percent")        or 0.0
        ram_percent        = m.get("ram_percent")        or 0.0

        client_data_list.append((
            client_id,
            training_time,
            communication_time,
            time_between_rounds,
            n_cpu,
            cpu_percent,
            ram_percent,
            model_type,
            data_distr,
            dataset_value,
            srt1,
            srt2
        ))

    # Scrivo le righe sul CSV
    num_clients = len(client_data_list)
    for idx, data in enumerate(client_data_list):
        (
            client_id,
            training_time,
            communication_time,
            tb_round,
            n_cpu,
            cpu_percent,
            ram_percent,
            model_type,
            data_distr,
            dataset_value,
            srt1,
            srt2
        ) = data
        last = (idx == num_clients - 1)
        log_round_time(
            client_id,
            currentRnd - 1,
            training_time,
            communication_time,
            tb_round,
            n_cpu,
            cpu_percent,
            ram_percent,
            model_type,
            data_distr,
            dataset_value,
            not last,
            srt1,
            srt2,
            agg_model_type
        )

    return {
        "train_loss":     avg_train_loss,
        "train_accuracy": avg_train_accuracy,
        "train_f1":       avg_train_f1,
        "train_mae":      avg_train_mae,
        "val_loss":       avg_val_loss,
        "val_accuracy":   avg_val_accuracy,
        "val_f1":         avg_val_f1,
        "val_mae":        avg_val_mae,
    }

parameters = ndarrays_to_parameters(get_weights_A(NetA()))
client_model_mapping = {}

class FedAvg(Strategy):
    def __init__(self, initial_parameters_a: Parameters):
        self.round_start_time: float | None = None
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
        self.round_start_time = time.time()
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
        global previous_round_end_time, currentRnd
        excluded_cid = []
        
        if CLIENT_SELECTOR and selection_strategy == "SSIM-Based":
            try:
                with open("exclusion_log.txt", "r") as f:
                    excluded_cid = f.read().strip()
                    log(INFO, f"[Round {currentRnd}] Escludo client {excluded_cid} dall'aggregazione")
            except FileNotFoundError:
                log(INFO, "PRIMO ROUND: nessun client da escludere")

        agg_start = time.time()
        round_total_time = time.time() - self.round_start_time
        log(INFO, f"Results Aggregated in {round_total_time:.2f} seconds.")

        results_a = []
        training_times = []
        #global currentRnd
        currentRnd += 1
        
        for client_proxy, fit_res in results:
            
            cid = fit_res.metrics.get("client_id") or client_proxy.cid
            client_id = fit_res.metrics.get("client_id")
            model_type = fit_res.metrics.get("model_type")
            training_time = fit_res.metrics.get("training_time")
            communication_time = fit_res.metrics.get("communication_time")
            compressed_parameters_hex = fit_res.metrics.get("compressed_parameters_hex")          
            client_model_mapping[client_id] = model_type

            if CLIENT_SELECTOR and selection_strategy == "SSIM-Based" and excluded_cid:
                if str(cid) == excluded_cid:
                    log(INFO, f"[Round {currentRnd}] Skipping aggregation of client {cid}")
                    continue

            if MESSAGE_COMPRESSOR:
                compressed_parameters = bytes.fromhex(compressed_parameters_hex)
                decompressed_parameters = pickle.loads(zlib.decompress(compressed_parameters))
                fit_res.parameters = ndarrays_to_parameters(decompressed_parameters)

            if training_time is not None:
                training_times.append(training_time)              
            results_a.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))

        previous_round_end_time = time.time()
        max_train = max(training_times) if training_times else 0.0

        agg_end = time.time()
        aggregation_time = agg_end - agg_start
        #log(INFO, f"Aggregation completed in {aggregation_time:.2f}s")

        self.parameters_a = self.aggregate_parameters(
            results_a,
            model_type,
            max_train,
            communication_time,
            round_total_time
        )

        aggregated_model = NetA()
        params_list = parameters_to_ndarrays(self.parameters_a)
        set_weights_A(aggregated_model, params_list)

        if MODEL_COVERSIONING:
            server_folder = os.path.join("model_weights", "server")
            os.makedirs(server_folder, exist_ok=True)
            path = os.path.join(server_folder, f"MW_round{currentRnd}.pt")
            torch.save(aggregated_model.state_dict(), path)
            log(INFO, f"Aggregated model weights saved to {path}")

        if currentRnd > 0 and CLIENT_SELECTOR and selection_strategy == "SSIM-Based":

            def get_image(path):
                with open(os.path.abspath(path), 'rb') as f:
                    with Image.open(f) as img:
                        return img.convert('RGB')

            def load_squeezenet_weights(model, weights_filename):
                state_dict = torch.load(weights_filename, weights_only=True)
                state_dict['classifier.1.weight'] = state_dict.pop('classifier.1.1.weight')
                state_dict['classifier.1.bias'] = state_dict.pop('classifier.1.1.bias')
                model.load_state_dict(state_dict)
                return model

            def load_shufflenet_weights(model, weights_filename):
                state_dict = torch.load(weights_filename, weights_only=True)
                state_dict['fc.weight'] = state_dict.pop('fc.1.weight')
                state_dict['fc.bias'] = state_dict.pop('fc.1.bias')
                model.load_state_dict(state_dict)
                return model

            def define_save_filename(weights_filename, method, image_filename):
                directory = os.path.dirname(weights_filename)
                image = os.path.splitext(os.path.basename(image_filename))[0]
                round = os.path.splitext(os.path.basename(weights_filename))[0]
                return f"{directory}/{method}_images/{method}_{image}_{round}.jpg"


            def run_cam(
                model: torch.nn.Module,
                target_layers: list,
                weights_filename: str,
                image_filename: str,
            ):
            
                img = get_image(image_filename)
                img = np.float32(img) / 255
                img_tensor = preprocess_image(
                    img,
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225],
                ).to("cpu")

                cam = GradCAM(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=img_tensor, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                img_explanation = show_cam_on_image(img, grayscale_cam, use_rgb=True)
                plt.imsave(define_save_filename(weights_filename, "gradcam", image_filename), img_explanation)
                return


            def compute_ssims(
                round: int = 1,
                model_weights_folder = "models/0-NoPatterns/shufflenet_v2_x0_5/model_weights/",
                model_name: str = "shufflenet_v2_x0_5",
            ):
                if model_name == "squeezenet1_1":
                    model = models.squeezenet1_1(weights=None, num_classes=10)
                    target_layers = [model.features[12].expand3x3]
                elif model_name == "shufflenet_v2_x0_5":
                    model = models.shufflenet_v2_x0_5(weights=None, num_classes=10)
                    target_layers = [model.conv5[0]]

                images = glob.glob("data/imagenet100-preprocessed/test/**/*.JPEG", recursive=True)
                server_pt = f"{model_weights_folder}/server/MW_round{round}.pt"
                client_dirs = sorted(glob.glob(os.path.join(model_weights_folder, "clients", "*")), key=lambda d: int(os.path.basename(d)))
                clients_pt = []
                client_ids = []
                for client_dir in client_dirs:
                    all_models = glob.glob(os.path.join(client_dir, "MW_round*.pt"))
                    valid = []
                    for p in all_models:
                        m = re.search(r"MW_round(\d+)\.pt$", p)
                        if m and int(m.group(1)) <= round:
                            valid.append((int(m.group(1)), p))
                    if valid:
                        latest_path = max(valid, key=lambda x: x[0])[1]
                        clients_pt.append(latest_path)
                        client_ids.append(int(os.path.basename(client_dir)))
                    else:
                        log(INFO, f"Nessun modello per client {os.path.basename(client_dir)} fino al round {round}.")

                if not os.path.exists(f"{os.path.dirname(server_pt)}/gradcam_images"):
                    os.makedirs(f"{os.path.dirname(server_pt)}/gradcam_images")

                for client_pt in clients_pt:
                    if not os.path.exists(f"{os.path.dirname(client_pt)}/gradcam_images"):
                        os.makedirs(f"{os.path.dirname(client_pt)}/gradcam_images")

                if model_name == "squeezenet1_1":
                    model = load_squeezenet_weights(model, server_pt)
                elif model_name == "shufflenet_v2_x0_5":
                    model = load_shufflenet_weights(model, server_pt)

                for image in images:
                    run_cam(
                        model=model,
                        target_layers=target_layers,
                        weights_filename=server_pt,
                        image_filename=image,
                    )

                for client_pt in clients_pt:
                    if model_name == "squeezenet1_1":
                        model = load_squeezenet_weights(model, client_pt)
                    elif model_name == "shufflenet_v2_x0_5":
                        model = load_shufflenet_weights(model, client_pt)

                    for image in images:
                        run_cam(
                            model=model,
                            target_layers=target_layers,
                            weights_filename=client_pt,
                            image_filename=image,
                        )
                ssim_values = []

                for client_pt in clients_pt:
                    client_ssim = []
                    for image in images:
                        server_image = get_image(define_save_filename(server_pt, "gradcam", image))
                        client_image = get_image(define_save_filename(client_pt, "gradcam", image))
                        server_image = img_as_float(server_image)
                        client_image = img_as_float(client_image)

                        ssim_value = ssim(server_image, client_image, data_range=server_image.max() - server_image.min(), channel_axis=2)
                        client_ssim.append(ssim_value)

                    client_ssim = np.array(client_ssim)
                    ssim_values.append(client_ssim.mean())

                return client_ids, ssim_values

            client_ids, ssim_values = compute_ssims(currentRnd, "model_weights", MODEL_NAME)

            values_str = ", ".join(f"Client {cid}: {val:.4f}"for cid, val in zip(client_ids, ssim_values))

            if selection_criteria.lower() == "mid":
                sorted_indices_and_values = sorted(enumerate(ssim_values), key=lambda x: x[1])
                n_clients = len(ssim_values)
                if n_clients % 2 == 1:
                    median_idx = n_clients // 2
                    exclude_idx = sorted_indices_and_values[median_idx][0]
                else:
                    median_idx_low = (n_clients // 2) - 1
                    exclude_idx = sorted_indices_and_values[median_idx_low][0]
            else:
                if selection_criteria.lower().startswith("max"):
                    exclude_idx = int(np.argmax(ssim_values))
                else:
                    exclude_idx = int(np.argmin(ssim_values))

            excluded_val = ssim_values[exclude_idx]
            excludingCID = exclude_idx + 1
            with open("exclusion_log.txt", "w") as f:
                f.write(f"{excludingCID}")

            log(
                INFO,
                f"\nRound {currentRnd} – {values_str}. "
                f"\nExcluding Client {exclude_idx+1} with SSIM={excluded_val:.4f}"
            )

        metrics_aggregated: Dict[str, Scalar] = {}
        if any(global_metrics[model_type].values()):
            metrics_aggregated[model_type] = {
                key: global_metrics[model_type][key][-1]
                if global_metrics[model_type][key]
                else None
                for key in global_metrics[model_type]
            }

        preprocess_csv()
        round_csv = os.path.join(
            performance_dir,
            f"FLwithAP_performance_metrics_round{currentRnd}.csv"
        )
        shutil.copy(csv_file, round_csv)

        return self.parameters_a, metrics_aggregated

    def aggregate_parameters(self, results, task_type, srt1, srt2, time_between_rounds):

        filtered = [
            (p, n, m)
            for p, n, m in results
            if n > 0
        ]
        total = sum(n for _, n, _ in filtered)
        new_weights = None
        metrics = []
        for params, n, m in filtered:
            w = n / total
            arrs = parameters_to_ndarrays(params)
            if new_weights is None:
                new_weights = [x * w for x in arrs]
            else:
                new_weights = [nw + x * w for nw, x in zip(new_weights, arrs)]
            metrics.append((n, m))

        weighted_average_global(metrics, task_type, srt1, srt2, time_between_rounds)

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
        return None

def server_fn(context: Context):

    strategy = FedAvg(
        initial_parameters_a=parameters,
    )
    server_config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    app.run(server_address="[::]:8080", config=ServerConfig(num_rounds=num_rounds))