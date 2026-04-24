import base64
from typing import List, Tuple, Dict, Optional
import os
import re
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, 
    AblationCAM, XGradCAM, EigenCAM, FullGrad
)
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from flwr.common import (
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
    ServerAppComponents
)
from io import BytesIO
from rich.panel import Panel
from flwr.server.strategy import Strategy, FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.common.logger import log
from logging import INFO
import textwrap
import numpy as np
from taskA import Net as NetA, get_weights as get_weights_A, set_weights as set_weights_A, load_data as load_data_A, normalize_dataset_name
from rich.console import Console
import shutil
import time
import csv
import os
import pandas as pd
import json  
import zlib
import pickle
import torch
from adaptation import AdaptationManager

# Silence progress bars printed by pytorch-grad-cam internals (ScoreCAM/AblationCAM).
_orig_tqdm = tqdm.tqdm
def _silent_tqdm(*args, **kwargs):
    kwargs.setdefault("disable", True)
    return _orig_tqdm(*args, **kwargs)
tqdm.tqdm = _silent_tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
folders_to_delete = ["performance", "model_weights"]

for folder in folders_to_delete:
    folder_path = os.path.join(current_dir, folder)
    if os.path.exists(folder_path):
        for nome in os.listdir(folder_path):
            percorso = os.path.join(folder_path, nome)
            if os.path.isdir(percorso):
                shutil.rmtree(percorso, ignore_errors=True)
            else:
                try:
                    os.remove(percorso)
                except OSError:
                    pass

################### GLOBAL PARAMETERS
global ADAPTATION
global metrics_history
metrics_history = {}
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MODEL_COVERSIONING, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MODEL_COVERSIONING = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False

global_metrics = {}
current_dir = os.path.abspath(os.path.dirname(__file__))

def config_patterns(config):
    global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MODEL_COVERSIONING, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER

    for pattern_name, pattern_info in config.items():
        if pattern_name == "client_selector":
            CLIENT_SELECTOR = pattern_info["enabled"]
        elif pattern_name == "client_cluster":
            CLIENT_CLUSTER = pattern_info["enabled"]
        elif pattern_name == "message_compressor":
            MESSAGE_COMPRESSOR = pattern_info["enabled"]
        elif pattern_name == "model_co-versioning_registry":
            MODEL_COVERSIONING = pattern_info["enabled"]
        elif pattern_name == "multi-task_model_trainer":
            MULTI_TASK_MODEL_TRAINER = pattern_info["enabled"]
        elif pattern_name == "heterogeneous_data_handler":
            HETEROGENEOUS_DATA_HANDLER = pattern_info["enabled"]

config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    ADAPTATION = config.get('adaptation', False).strip().lower()

    num_rounds = int(os.environ.get("AP4FED_ROUNDS_OVERRIDE", config.get('rounds', 10)))
    client_count = int(config.get('clients', 2))
    clients_per_round = int(config.get('clients_per_round', client_count))
    config_patterns(config["patterns"])

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
            "data_persistence_type": client.get("data_persistence_type"),
            "model": client.get("model"),
            "epochs": client.get("epochs"),
        })
    GLOBAL_CLIENT_DETAILS = client_details_structure

selector_params = config["patterns"]["client_selector"]["params"]
selection_strategy = selector_params.get("selection_strategy")      
selection_criteria = selector_params.get("selection_criteria")
explainer_type = selector_params.get("explainer_type", "GradCAM")
MODEL_NAME = GLOBAL_CLIENT_DETAILS[0]["model"]
DATASET_NAME = normalize_dataset_name(config.get("dataset") or GLOBAL_CLIENT_DETAILS[0].get("dataset"))
currentRnd = 0
ssim_overhead_by_round = {}

performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

performance_ml_dir = './performance_MLdata/'
if not os.path.exists(performance_ml_dir):
    os.makedirs(performance_ml_dir)

csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)

ml_csv_file = os.path.join(performance_ml_dir, 'FLwithAP_MLdata.csv')

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Client ID', 'FL Round',
        'Training Time', 'JSD', 'HDH Time', 'Communication Time', 'Total Time of FL Round',
        '# of CPU', 'CPU Usage (%)', 'RAM Usage (%)',
        'Model Type', 'Data Distr. Type', 'Dataset',
        'Train Loss', 'Train Accuracy', 'Train F1', 'Train MAE',
        'Val Loss', 'Val Accuracy', 'Val F1', 'Val MAE',
        'SSIM Overhead (indice)',
        'AP List (client_selector,client_cluster,message_compressor,model_co-versioning_registry,multi-task_model_trainer,heterogeneous_data_handler)'
    ])

def log_round_time(
        client_id, fl_round,
        training_time, jsd, hdh_ms, communication_time, time_between_rounds,
        n_cpu, cpu_percent, ram_percent,
        client_model_type, data_distr, dataset_value,
        already_logged, srt1, srt2, agg_key, ssim_overhead=None
):

    if agg_key not in global_metrics:
        global_metrics[agg_key] = {
            "train_loss": [], "train_accuracy": [], "train_f1": [], "train_mae": [],
            "val_loss": [], "val_accuracy": [], "val_f1": [], "val_mae": []
        }

    tm = global_metrics[agg_key]
    train_loss = tm["train_loss"][-1] if tm["train_loss"] else None
    train_accuracy = tm["train_accuracy"][-1] if tm["train_accuracy"] else None
    train_f1 = tm["train_f1"][-1] if tm["train_f1"] else None
    train_mae = tm["train_mae"][-1] if tm["train_mae"] else None
    val_loss = tm["val_loss"][-1] if tm["val_loss"] else None
    val_accuracy = tm["val_accuracy"][-1] if tm["val_accuracy"] else None
    val_f1 = tm["val_f1"][-1] if tm["val_f1"] else None
    val_mae = tm["val_mae"][-1] if tm["val_mae"] else None

    if already_logged:
        srt2 = None
        ssim_overhead = None

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        def onoff(b):
            return "ON" if b else "OFF"

        ap_list = "{" + ",".join([
            onoff(CLIENT_SELECTOR),
            onoff(CLIENT_CLUSTER),
            onoff(MESSAGE_COMPRESSOR),
            onoff(MODEL_COVERSIONING),
            onoff(MULTI_TASK_MODEL_TRAINER),
            onoff(HETEROGENEOUS_DATA_HANDLER),
        ]) + "}"

        writer.writerow([
            client_id,
            fl_round + 1,
            f"{training_time:.2f}",
            f"{jsd:.2f}",
            f"{hdh_ms:.2f}",
            f"{communication_time:.2f}",
            f"{time_between_rounds:.2f}",
            n_cpu,
            f"{cpu_percent:.0f}",
            f"{ram_percent:.0f}",
            client_model_type,
            data_distr,
            dataset_value,
            f"{train_loss:.2f}" if train_loss is not None else "",
            f"{train_accuracy:.4f}" if train_accuracy is not None else "",
            f"{train_f1:.4f}" if train_f1 is not None else "",
            f"{train_mae:.4f}" if train_mae is not None else "",
            f"{val_loss:.2f}" if val_loss is not None else "",
            f"{val_accuracy:.4f}" if val_accuracy is not None else "",
            f"{val_f1:.4f}" if val_f1 is not None else "",
            f"{val_mae:.4f}" if val_mae is not None else "",
            f"{ssim_overhead:.2f}" if ssim_overhead is not None else "",
            ap_list
        ])


def preprocess_csv(agent_time=None):
    import pandas as pd
    df = pd.read_csv(csv_file)

    df["Client Number"] = (
        df["Client ID"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    df["Training Time"] = pd.to_numeric(df["Training Time"], errors="coerce")
    df["JSD"] = pd.to_numeric(df["JSD"], errors="coerce")

    df["Total Time of FL Round"] = pd.to_numeric(
        df["Total Time of FL Round"], errors="coerce"
    )

    df["Total Time of FL Round"] = (
        df.groupby("FL Round")["Total Time of FL Round"]
        .transform(lambda x: [None] * (len(x) - 1) + [x.iloc[-1]])
    )

    if "SSIM Overhead (indice)" not in df.columns:
        df["SSIM Overhead (indice)"] = pd.NA
    else:
        df["SSIM Overhead (indice)"] = pd.to_numeric(df["SSIM Overhead (indice)"], errors="coerce")

    df.sort_values(["FL Round", "Client Number"], inplace=True)
    cols_round = [
        c for c in
        ["Total Time of FL Round"] + list(df.columns[df.columns.get_loc("Train Loss"):])
        if c not in ("FL Round", "Client Number")
    ]

    def fix_round_values(subdf):
        subdf = subdf.copy()
        last = subdf["Client Number"].max()
        for col in cols_round:
            if col not in subdf.columns:
                continue
            vals = subdf[col].dropna()
            v = vals.iloc[-1] if not vals.empty else pd.NA
            subdf.loc[subdf["Client Number"] == last, col] = v
            subdf.loc[subdf["Client Number"] != last, col] = pd.NA
        return subdf

    fl_round_backup = df["FL Round"].copy()
    gb = df.groupby("FL Round", group_keys=False)

    try:
        df = gb.apply(fix_round_values, include_groups=False)
    except TypeError:
        df = gb.apply(fix_round_values)

    if "FL Round" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "FL Round" in df.index.names:
            df = df.reset_index(level="FL Round")
        elif getattr(df.index, "name", None) == "FL Round":
            df = df.reset_index()
        else:
            insert_pos = list(df.columns).index("Client ID") + 1 if "Client ID" in df.columns else 0
            df.insert(insert_pos, "FL Round", fl_round_backup)

    for fl_round, overhead in ssim_overhead_by_round.items():
        mask = df["FL Round"] == fl_round
        idx = df[mask].tail(1).index
        if len(idx) > 0:
            df.loc[idx, "SSIM Overhead (indice)"] = round(float(overhead), 2)

    if 'Agent Time (s)' not in df.columns:
        df['Agent Time (s)'] = 0.0
    else:
        df['Agent Time (s)'] = pd.to_numeric(df['Agent Time (s)'], errors='coerce').fillna(0.0)
    current_round = df['FL Round'].max()
    mask = df['FL Round'] == current_round
    idx = df[mask].tail(1).index
    if 'Agent Time (s)' not in df.columns:
        df['Agent Time (s)'] = np.nan
    try:
        val = round(float(agent_time), 2)
    except (TypeError, ValueError):
        val = np.nan
    df.loc[idx, 'Agent Time (s)'] = val
    df.drop(columns=["Client Number"], inplace=True)
    df.to_csv(csv_file, index=False)


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _serialize_value(value):
    if value is None:
        return np.nan
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _nan_if_missing(value):
    if value is None:
        return np.nan
    if isinstance(value, str) and not value.strip():
        return np.nan
    return value


def _pattern_state(enabled):
    return "ON" if enabled else "OFF"


def _build_ap_list_for_ml():
    return "{" + ",".join([
        _pattern_state(CLIENT_SELECTOR),
        _pattern_state(MESSAGE_COMPRESSOR),
        _pattern_state(HETEROGENEOUS_DATA_HANDLER),
    ]) + "}"


def _infer_alpha_dirichlet(client_detail):
    distribution = str(client_detail.get("data_distribution_type", "")).strip().lower()
    if distribution == "iid":
        return 1
    if distribution == "non-iid":
        return client_detail.get("non_iid_alpha", client_detail.get("alpha_dirichlet", 0.5))
    if "non_iid_alpha" in client_detail:
        return client_detail.get("non_iid_alpha")
    if "alpha_dirichlet" in client_detail:
        return client_detail.get("alpha_dirichlet")
    if client_detail.get("data_persistence_type") in {"New Data", "Remove Data"}:
        return 0.5
    return np.nan


def _resolve_checkpoint_round(available_rounds, fraction):
    if not available_rounds:
        return None
    target_round = int(np.ceil(num_rounds * fraction))
    target_round = max(1, min(num_rounds, target_round))
    if target_round in available_rounds:
        return target_round
    lower_or_equal = [rnd for rnd in available_rounds if rnd <= target_round]
    if lower_or_equal:
        return max(lower_or_equal)
    return min(available_rounds)


def _collect_checkpoint_metrics(df, round_level_df, round_number):
    if round_number is None:
        return {}
    round_df = df[df["FL Round"] == round_number]
    round_level_row = round_level_df[round_level_df["FL Round"] == round_number]
    if round_df.empty or round_level_row.empty:
        return {}
    last_row = round_level_row.iloc[-1]
    return {
        "round": int(round_number),
        "val_f1": _safe_float(last_row.get("Val F1")),
        "training_time_avg": _safe_float(round_df["Training Time"].mean()),
        "communication_time_avg": _safe_float(round_df["Communication Time"].mean()),
        "total_round_time": _safe_float(last_row.get("Total Time of FL Round")),
    }


def build_ml_experiment_row(report_path):
    def _build_empty_row():
        first_client = GLOBAL_CLIENT_DETAILS[0] if GLOBAL_CLIENT_DETAILS else {}
        first_model = first_client.get("model", "")
        first_dataset = config.get("dataset") or first_client.get("dataset", "")
        unique_epochs = sorted(
            {
                int(client.get("epochs"))
                for client in GLOBAL_CLIENT_DETAILS
                if client.get("epochs") is not None
            }
        )
        epochs_value = unique_epochs[0] if len(unique_epochs) == 1 else _serialize_value(unique_epochs)
        selector_block = config.get("patterns", {}).get("client_selector", {})
        selector_cfg = selector_block.get("params", {})
        selector_enabled = bool(selector_block.get("enabled", False))
        message_compressor_enabled = bool(config.get("patterns", {}).get("message_compressor", {}).get("enabled", False))
        hdh_enabled = bool(config.get("patterns", {}).get("heterogeneous_data_handler", {}).get("enabled", False))

        row = {
            "N Rounds": int(num_rounds),
            "Total Clients": int(client_count),
            "Model": first_model,
            "Dataset": first_dataset,
            "Optimizer": "Adam",
            "Learning Rate": 0.001,
            "Batch Size": 64,
            "Epochs": epochs_value,
        }

        for client_detail in sorted(GLOBAL_CLIENT_DETAILS, key=lambda item: int(item.get("client_id", 0))):
            client_number = int(client_detail.get("client_id", 0))
            prefix = f"Client {client_number}"
            row[f"{prefix} ID"] = prefix
            row[f"{prefix} CPU"] = client_detail.get("cpu", "")
            row[f"{prefix} RAM"] = client_detail.get("ram", "")
            row[f"{prefix} Data Distribution"] = _nan_if_missing(client_detail.get("data_distribution_type"))
            row[f"{prefix} Data Persistence"] = _nan_if_missing(client_detail.get("data_persistence_type"))
            row[f"{prefix} Alpha Dirichlet"] = _infer_alpha_dirichlet(client_detail)
            row[f"{prefix} JSD"] = np.nan
            row[f"{prefix} CPU Usage Avg"] = np.nan
            row[f"{prefix} RAM Usage Avg"] = np.nan

        for label in ("25", "50", "75"):
            row[f"Round {label}%"] = np.nan
            row[f"Val F1 {label}%"] = np.nan
            row[f"Training Time Avg {label}%"] = np.nan
            row[f"Communication Time Avg {label}%"] = np.nan
            row[f"Total Round Time {label}%"] = np.nan

        row.update({
            "Avg Training Time": np.nan,
            "Avg Communication Time": np.nan,
            "Avg Total Round Time": np.nan,
            "Final Train Loss": np.nan,
            "Final Train Accuracy": np.nan,
            "Final Train F1": np.nan,
            "Final Val Loss": np.nan,
            "Final Val Accuracy": np.nan,
            "Final Val F1 (Last Round)": np.nan,
            "Final Val F1 (Best)": np.nan,
            "Final Agent Time": np.nan,
            "AP List (client_selector,message_compressor,heterogeneous_data_handler)": _build_ap_list_for_ml(),
            "Client Selector Strategy": _nan_if_missing(selector_cfg.get("selection_strategy")) if selector_enabled else np.nan,
            "Client Selector Criteria": _nan_if_missing(selector_cfg.get("selection_criteria")) if selector_enabled else np.nan,
            "Client Selector Value": _nan_if_missing(selector_cfg.get("selection_value")) if selector_enabled else np.nan,
            "Message Compressor Alg": "zlib" if message_compressor_enabled else np.nan,
            "HDH Batch Size": 32 if hdh_enabled else np.nan,
            "HDH Beta 1": 0.5 if hdh_enabled else np.nan,
            "HDH Beta 2": 0.999 if hdh_enabled else np.nan,
            "HDH Discriminator": "DCGANDiscriminator" if hdh_enabled else np.nan,
            "HDH Epochs": 1 if hdh_enabled else np.nan,
            "HDH Generator": "DCGANGenerator" if hdh_enabled else np.nan,
            "HDH Learning Rate": 2e-4 if hdh_enabled else np.nan,
            "HDH Latent Dim": 100 if hdh_enabled else np.nan,
            "HDH Optimizer": "Adam" if hdh_enabled else np.nan,
        })
        return row

    if not os.path.exists(report_path):
        return _build_empty_row()

    df = pd.read_csv(report_path)
    if df.empty:
        return _build_empty_row()

    numeric_cols = [
        "FL Round",
        "Training Time",
        "JSD",
        "HDH Time",
        "Communication Time",
        "Total Time of FL Round",
        "# of CPU",
        "CPU Usage (%)",
        "RAM Usage (%)",
        "Train Loss",
        "Train Accuracy",
        "Train F1",
        "Train MAE",
        "Val Loss",
        "Val Accuracy",
        "Val F1",
        "Val MAE",
        "SSIM Overhead (indice)",
        "Agent Time (s)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Client Number"] = (
        df["Client ID"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )
    df.sort_values(["FL Round", "Client Number"], inplace=True)
    round_level_df = df.groupby("FL Round", group_keys=False).tail(1).copy()
    round_level_df.sort_values("FL Round", inplace=True)
    available_rounds = round_level_df["FL Round"].dropna().astype(int).tolist()
    if not available_rounds:
        return _build_empty_row()

    first_client = GLOBAL_CLIENT_DETAILS[0] if GLOBAL_CLIENT_DETAILS else {}
    first_model = first_client.get("model", "")
    first_dataset = config.get("dataset") or first_client.get("dataset", "")
    unique_epochs = sorted(
        {
            int(client.get("epochs"))
            for client in GLOBAL_CLIENT_DETAILS
            if client.get("epochs") is not None
        }
    )
    epochs_value = unique_epochs[0] if len(unique_epochs) == 1 else _serialize_value(unique_epochs)
    selector_block = config.get("patterns", {}).get("client_selector", {})
    selector_cfg = selector_block.get("params", {})
    selector_enabled = bool(selector_block.get("enabled", False))
    message_compressor_enabled = bool(config.get("patterns", {}).get("message_compressor", {}).get("enabled", False))
    hdh_enabled = bool(config.get("patterns", {}).get("heterogeneous_data_handler", {}).get("enabled", False))
    final_row = round_level_df.iloc[-1]

    row = {
        "N Rounds": int(num_rounds),
        "Total Clients": int(client_count),
        "Model": first_model,
        "Dataset": first_dataset,
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 64,
        "Epochs": epochs_value,
    }

    for client_detail in sorted(GLOBAL_CLIENT_DETAILS, key=lambda item: int(item.get("client_id", 0))):
        client_number = int(client_detail.get("client_id", 0))
        prefix = f"Client {client_number}"
        client_df = df[df["Client Number"] == client_number]
        row[f"{prefix} ID"] = prefix
        row[f"{prefix} CPU"] = client_detail.get("cpu", "")
        row[f"{prefix} RAM"] = client_detail.get("ram", "")
        row[f"{prefix} Data Distribution"] = _nan_if_missing(client_detail.get("data_distribution_type"))
        row[f"{prefix} Data Persistence"] = _nan_if_missing(client_detail.get("data_persistence_type"))
        row[f"{prefix} Alpha Dirichlet"] = _infer_alpha_dirichlet(client_detail)
        row[f"{prefix} JSD"] = _safe_float(client_df["JSD"].mean()) if (hdh_enabled and not client_df.empty) else np.nan
        row[f"{prefix} CPU Usage Avg"] = _safe_float(client_df["CPU Usage (%)"].mean()) if not client_df.empty else np.nan
        row[f"{prefix} RAM Usage Avg"] = _safe_float(client_df["RAM Usage (%)"].mean()) if not client_df.empty else np.nan

    for fraction, label in [(0.25, "25"), (0.50, "50"), (0.75, "75")]:
        checkpoint_round = _resolve_checkpoint_round(available_rounds, fraction)
        checkpoint = _collect_checkpoint_metrics(df, round_level_df, checkpoint_round)
        row[f"Round {label}%"] = checkpoint.get("round", "")
        row[f"Val F1 {label}%"] = checkpoint.get("val_f1")
        row[f"Training Time Avg {label}%"] = checkpoint.get("training_time_avg")
        row[f"Communication Time Avg {label}%"] = checkpoint.get("communication_time_avg")
        row[f"Total Round Time {label}%"] = checkpoint.get("total_round_time")

    row.update({
        "Avg Training Time": _safe_float(df["Training Time"].mean()),
        "Avg Communication Time": _safe_float(df["Communication Time"].mean()),
        "Avg Total Round Time": _safe_float(round_level_df["Total Time of FL Round"].mean()),
        "Final Train Loss": _safe_float(final_row.get("Train Loss")),
        "Final Train Accuracy": _safe_float(final_row.get("Train Accuracy")),
        "Final Train F1": _safe_float(final_row.get("Train F1")),
        "Final Val Loss": _safe_float(final_row.get("Val Loss")),
        "Final Val Accuracy": _safe_float(final_row.get("Val Accuracy")),
        "Final Val F1 (Last Round)": _safe_float(final_row.get("Val F1")),
        "Final Val F1 (Best)": _safe_float(round_level_df["Val F1"].max()),
        "Final Agent Time": _safe_float(final_row.get("Agent Time (s)")) if "Agent Time (s)" in round_level_df.columns else None,
        "AP List (client_selector,message_compressor,heterogeneous_data_handler)": _build_ap_list_for_ml(),
        "Client Selector Strategy": _nan_if_missing(selector_cfg.get("selection_strategy")) if selector_enabled else np.nan,
        "Client Selector Criteria": _nan_if_missing(selector_cfg.get("selection_criteria")) if selector_enabled else np.nan,
        "Client Selector Value": _nan_if_missing(selector_cfg.get("selection_value")) if selector_enabled else np.nan,
        "Message Compressor Alg": "zlib" if message_compressor_enabled else np.nan,
        "HDH Batch Size": 32 if hdh_enabled else np.nan,
        "HDH Beta 1": 0.5 if hdh_enabled else np.nan,
        "HDH Beta 2": 0.999 if hdh_enabled else np.nan,
        "HDH Discriminator": "DCGANDiscriminator" if hdh_enabled else np.nan,
        "HDH Epochs": 1 if hdh_enabled else np.nan,
        "HDH Generator": "DCGANGenerator" if hdh_enabled else np.nan,
        "HDH Learning Rate": 2e-4 if hdh_enabled else np.nan,
        "HDH Latent Dim": 100 if hdh_enabled else np.nan,
        "HDH Optimizer": "Adam" if hdh_enabled else np.nan,
    })

    return row


def append_ml_experiment_row(row):
    if not row:
        return
    row_df = pd.DataFrame([row])
    ordered_columns = list(row_df.columns)
    if os.path.exists(ml_csv_file):
        try:
            existing_df = pd.read_csv(ml_csv_file, sep=";", decimal=",")
        except Exception:
            existing_df = pd.read_csv(ml_csv_file)
        combined_df = pd.concat([existing_df, row_df], ignore_index=True, sort=False)
        ordered_columns.extend([col for col in combined_df.columns if col not in ordered_columns])
        combined_df = combined_df.reindex(columns=ordered_columns)
        combined_df.to_csv(ml_csv_file, index=False, na_rep="NaN", sep=";", decimal=",")
    else:
        row_df.to_csv(ml_csv_file, index=False, na_rep="NaN", sep=";", decimal=",")

def weighted_average_global(metrics, agg_model_type, srt1, srt2, time_between_rounds, ssim_overhead=None):
    if agg_model_type not in global_metrics:
        global_metrics[agg_model_type] = {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "train_mae": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_mae": [],
            "jsd": []
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

    train_losses = [n * (m.get("train_loss") or 0) for n, m in metrics]
    train_accuracies = [n * (m.get("train_accuracy") or 0) for n, m in metrics]
    train_f1 = [n * (m.get("train_f1") or 0) for n, m in metrics]
    train_maes = [n * (m.get("train_mae") or 0) for n, m in metrics]
    val_losses = [n * (m.get("val_loss") or 0) for n, m in metrics]
    val_accuracies = [n * (m.get("val_accuracy") or 0) for n, m in metrics]
    val_f1 = [n * (m.get("val_f1") or 0) for n, m in metrics]
    val_maes = [n * (m.get("val_mae") or 0) for n, m in metrics]
    jsds = sorted([(m.get("client_id"), m.get("jsd") or 0) for _, m in metrics],
                  key=lambda x: x[0])

    avg_train_loss = sum(train_losses) / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_train_f1 = sum(train_f1) / total_examples
    avg_train_mae = sum(train_maes) / total_examples
    avg_val_loss = sum(val_losses) / total_examples
    avg_val_accuracy = sum(val_accuracies) / total_examples
    avg_val_f1 = sum(val_f1) / total_examples
    avg_val_mae = sum(val_maes) / total_examples
    sorted_jsds = tuple([tup[1] for tup in jsds])

    global_metrics[agg_model_type]["train_loss"].append(avg_train_loss)
    global_metrics[agg_model_type]["train_accuracy"].append(avg_train_accuracy)
    global_metrics[agg_model_type]["train_f1"].append(avg_train_f1)
    global_metrics[agg_model_type]["train_mae"].append(avg_train_mae)
    global_metrics[agg_model_type]["val_loss"].append(avg_val_loss)
    global_metrics[agg_model_type]["val_accuracy"].append(avg_val_accuracy)
    global_metrics[agg_model_type]["val_f1"].append(avg_val_f1)
    global_metrics[agg_model_type]["val_mae"].append(avg_val_mae)
    global_metrics[agg_model_type]["jsd"].append(sorted_jsds)

    client_data_list = []
    for num_examples, m in metrics:
        if num_examples == 0:
            continue
        client_id = m.get("client_id")
        model_type = m.get("model_type", "N/A")
        data_distr = m.get("data_distribution_type", "N/A")
        dataset_value = m.get("dataset", "N/A")
        training_time = m.get("training_time") or 0.0
        jsd = m.get("jsd") or 0.0
        communication_time = m.get("communication_time") or 0.0
        n_cpu = m.get("n_cpu") or 0
        hdh_ms = m.get("hdh_ms") or 0.0
        cpu_percent = m.get("cpu_percent") or 0.0
        ram_percent = m.get("ram_percent") or 0.0
        if client_id:
            client_data_list.append((
                client_id,
                training_time,
                jsd,
                hdh_ms,
                communication_time,
                time_between_rounds,
                n_cpu,
                cpu_percent,
                ram_percent,
                model_type,
                data_distr,
                dataset_value,
                srt1,
                srt2,
                ssim_overhead
            ))

    num_clients = len(client_data_list)
    for idx, client_data in enumerate(client_data_list):
        (
            client_id,
            training_time,
            jsd,
            hdh_ms,
            communication_time,
            time_between_rounds,
            n_cpu,
            cpu_percent,
            ram_percent,
            model_type,
            data_distr,
            dataset_value,
            srt1,
            srt2,
            ssim_overhead
        ) = client_data
        already_logged = (idx != num_clients - 1)
        log_round_time(
            client_id,
            currentRnd - 1,
            training_time,
            jsd,
            hdh_ms,
            communication_time,
            time_between_rounds,
            n_cpu,
            cpu_percent,
            ram_percent,
            model_type,
            data_distr,
            dataset_value,
            already_logged,
            srt1,
            srt2,
            agg_model_type,
            ssim_overhead
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
        "jsd": sorted_jsds
    }

parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
client_model_mapping = {}


class ClientCidCriterion(Criterion):
    def __init__(self, allowed_cids):
        self.allowed_cids = {str(cid) for cid in allowed_cids}

    def select(self, client: ClientProxy) -> bool:
        return str(client.cid) in self.allowed_cids


def _client_detail_by_cid(cid: str):
    cid_str = str(cid)
    for detail in GLOBAL_CLIENT_DETAILS:
        if str(detail.get("client_id")) == cid_str:
            return detail
    return {}


def _is_resource_eligible_client(cid: str) -> bool:
    if not (CLIENT_SELECTOR and selection_strategy == "Resource-Based"):
        return True
    detail = _client_detail_by_cid(cid)
    if not detail:
        return True
    selection_value = int(selector_params.get("selection_value", 0) or 0)
    if selection_criteria == "CPU":
        return int(detail.get("cpu", 0) or 0) >= selection_value
    if selection_criteria == "RAM":
        return int(detail.get("ram", 0) or 0) >= selection_value
    return True


def _is_high_tier_client(cid: str) -> bool:
    detail = _client_detail_by_cid(cid)
    if not detail:
        return False
    selection_value = int(selector_params.get("selection_value", 0) or 0)
    if selection_criteria == "CPU":
        return int(detail.get("cpu", 0) or 0) > selection_value
    if selection_criteria == "RAM":
        return int(detail.get("ram", 0) or 0) > selection_value
    return False

class FedAvg(Strategy):
    def __init__(self, initial_parameters_a: Parameters):
        self.round_start_time: float | None = None
        self.parameters_a = initial_parameters_a
        self._send_ts = {}
        self.excluded_cid = None
        self._used_training_client_cids = set()
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

        if ADAPTATION == "none" or not ADAPTATION:
            log(INFO, "Adaptation Disabled ❌")
            self.adapt_mgr = AdaptationManager(False, config)
        else:
            pol_display = ADAPTATION.title()
            if "voting" in ADAPTATION:
                pol_display = "Voting-Based"
            elif "role" in ADAPTATION:
                pol_display = "Role-Based"
            elif "debate" in ADAPTATION:
                pol_display = "Debate-Based"
            elif "expert" in ADAPTATION:
                pol_display = "Expert-Driven"
            elif "random" in ADAPTATION:
                pol_display = "Random"
            elif "single" in ADAPTATION or "ai-agents" in ADAPTATION:
                pol_display = "AI-Agents"
                
            log(INFO, f"Adaptation Enabled ✅ - {pol_display}")
            self.adapt_mgr = AdaptationManager(True, config)
            self.adapt_mgr.describe()

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
        client_manager.wait_for(client_count)
        self.round_start_time = time.time()
        available = client_manager.num_available()
        if available < 1:
            return []

        num_fit = min(available, max(1, clients_per_round))
        all_clients = client_manager.all()
        unseen_cids = [
            cid for cid in all_clients.keys()
            if str(cid) not in self._used_training_client_cids
        ]

        clients: List[ClientProxy] = []
        if unseen_cids:
            clients.extend(
                client_manager.sample(
                    num_clients=min(num_fit, len(unseen_cids)),
                    min_num_clients=1,
                    criterion=ClientCidCriterion(unseen_cids),
                )
            )

        if len(clients) < num_fit:
            selected_cids = {str(client.cid) for client in clients}
            remaining_cids = [
                cid for cid in all_clients.keys()
                if str(cid) not in selected_cids
            ]
            remaining_needed = num_fit - len(clients)
            if remaining_cids and remaining_needed > 0:
                clients.extend(
                    client_manager.sample(
                        num_clients=min(remaining_needed, len(remaining_cids)),
                        min_num_clients=1,
                        criterion=ClientCidCriterion(remaining_cids),
                    )
                )

        if CLIENT_SELECTOR and selection_strategy == "Resource-Based":
            eligible_clients = [client for client in clients if _is_resource_eligible_client(client.cid)]
            if len(eligible_clients) < 2:
                selected_cids = {str(client.cid) for client in clients}
                preferred_high_tier_cids = [
                    cid for cid in self._used_training_client_cids
                    if cid in all_clients
                    and cid not in selected_cids
                    and _is_high_tier_client(cid)
                ]
                needed_high_tier = 2 - len(eligible_clients)
                candidate_high_tier_cids = preferred_high_tier_cids[:]
                if needed_high_tier > 0 and len(candidate_high_tier_cids) < needed_high_tier:
                    global_high_tier_cids = [
                        cid for cid in all_clients.keys()
                        if str(cid) not in selected_cids
                        and str(cid) not in candidate_high_tier_cids
                        and _is_high_tier_client(cid)
                    ]
                    candidate_high_tier_cids.extend(global_high_tier_cids)
                if candidate_high_tier_cids and needed_high_tier > 0:
                    fallback_clients = client_manager.sample(
                        num_clients=min(needed_high_tier, len(candidate_high_tier_cids)),
                        min_num_clients=1,
                        criterion=ClientCidCriterion(candidate_high_tier_cids),
                    )
                    fallback_by_cid = {str(client.cid): client for client in fallback_clients}
                    updated_clients = list(clients)
                    replaceable_indexes = [
                        idx for idx, client in enumerate(updated_clients)
                        if not _is_resource_eligible_client(client.cid)
                    ]
                    for replace_idx, fallback_client in zip(replaceable_indexes, fallback_by_cid.values()):
                        updated_clients[replace_idx] = fallback_client
                    deduped_clients = []
                    seen_final_cids = set()
                    for client in updated_clients:
                        cid = str(client.cid)
                        if cid not in seen_final_cids:
                            deduped_clients.append(client)
                            seen_final_cids.add(cid)
                    clients = deduped_clients

        self._used_training_client_cids.update(str(client.cid) for client in clients)

        base_params: Parameters = self.parameters_a if self.parameters_a is not None else parameters
        if base_params is None:
            return []

        fake_parameters = Parameters(tensors=[], tensor_type=base_params.tensor_type)
        blob = pickle.dumps(base_params, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(blob, level=1)
        compressed_parameters_b64 = base64.b64encode(compressed).decode("ascii")
        fit_configurations: List[Tuple[ClientProxy, FitIns]] = []
        for client in clients:
            self._send_ts[client.cid] = time.time()
            if 'MESSAGE_COMPRESSOR' in globals() and MESSAGE_COMPRESSOR:
                cfg = {"compressed_parameters_b64": compressed_parameters_b64}
                fit_ins = FitIns(fake_parameters, cfg)
            else:
                fit_ins = FitIns(base_params, {})

            fit_configurations.append((client, fit_ins))

        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        global previous_round_end_time, currentRnd

        if CLIENT_SELECTOR and selection_strategy == "SSIM-Based":
            if self.excluded_cid:
                log(INFO, f"[Round {currentRnd}] Client {self.excluded_cid} excluded from aggregation")
            else:
                log(INFO, f"[Round {currentRnd}] No clients excluded from previous round")

        agg_start = time.time()
        round_total_time = time.time() - self.round_start_time
        log(INFO, f"Results Aggregated in {round_total_time:.2f} seconds.")

        results_a = []
        training_times = []
        currentRnd += 1

        for client_proxy, fit_res in results:
            compressed_parameters_b64 = None
            if fit_res.num_examples == 0:
                training_time = None
                communication_time = None
                compressed_parameters_hex = None
                client_id = client_proxy.cid
                model_type = None
                metrics = {
                    "train_loss": None,
                    "train_accuracy": None,
                    "train_f1": None,
                    "train_mae": None,
                    "val_loss": None,
                    "val_accuracy": None,
                    "val_f1": None,
                    "val_mae": None,
                    "training_time": None,
                    "communication_time": None,
                    "compressed_parameters_hex": None,
                    "client_id": client_id,
                    "model_type": None,
                }
            else:
                metrics = fit_res.metrics or {}
                training_time = metrics.get("training_time")
                communication_time = metrics.get("communication_time")
                compressed_parameters_b64 = metrics.get("compressed_parameters_b64")
                client_id = metrics.get("client_id")
                model_type = metrics.get("model_type")
                hdh_ms = metrics.get("hdh_ms", 0.0)
                client_model_mapping[client_id] = model_type
                recv_ts = metrics.get("client_sent_ts", None) or time.time()
                send_ts = self._send_ts.get(client_proxy.cid, recv_ts)
                rt_total = recv_ts - send_ts
                train_t = training_time or 0.0
                server_comm_time = max(rt_total - train_t - hdh_ms, 0.0)
                if metrics.get("communication_time") is None:
                    metrics["communication_time"] = server_comm_time
                else:
                    metrics["server_comm_time"] = server_comm_time

            if CLIENT_SELECTOR and selection_strategy == "SSIM-Based" and self.excluded_cid:
                if str(client_id) == str(self.excluded_cid):
                    log(INFO, f"[Round {currentRnd}] Skipping aggregation of client {client_id}")
                    continue

            if MESSAGE_COMPRESSOR and compressed_parameters_b64:
                compressed = base64.b64decode(compressed_parameters_b64)
                decompressed = pickle.loads(zlib.decompress(compressed))
                fit_res.parameters = ndarrays_to_parameters(decompressed)

            if training_time is not None:
                training_times.append(training_time)

            results_a.append((fit_res.parameters, fit_res.num_examples, metrics))

        effective_results_a = [
            item for item in results_a
            if item[1] and item[1] > 0
        ]

        if not effective_results_a:
            previous_round_end_time = time.time()
            log(
                INFO,
                f"[Round {currentRnd}] No clients satisfied the selector criteria for aggregation. Skipping round.",
            )
            return self.parameters_a, {}

        previous_round_end_time = time.time()
        max_train = max(training_times) if training_times else 0.0
        agg_end = time.time()
        aggregation_time = agg_end - agg_start

        self.parameters_a = self.aggregate_parameters(
            effective_results_a,
            model_type,
            max_train,
            communication_time,
            round_total_time
        )

        aggregated_model = NetA()
        params_list = parameters_to_ndarrays(self.parameters_a)
        set_weights_A(aggregated_model, params_list)

        if MODEL_COVERSIONING or (CLIENT_SELECTOR and selection_strategy == "SSIM-Based"):
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

            def define_save_filename(weights_filename, method, image_key):
                directory = os.path.dirname(weights_filename)
                image = str(image_key)
                round = os.path.splitext(os.path.basename(weights_filename))[0]
                return f"{directory}/{method}_images/{method}_{image}_{round}.jpg"


            def run_cam(
                model: torch.nn.Module,
                target_layers: list,
                weights_filename: str,
                image_array: np.ndarray,
                image_key: str,
                explainer_type: str = "GradCAM",
            ):
                img = np.float32(image_array) / 255.0
                img_tensor = preprocess_image(
                    img,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ).to("cpu")

                # Map string to class
                cam_methods = {
                    "GradCAM": GradCAM,
                    "HiResCAM": HiResCAM,
                    "ScoreCAM": ScoreCAM,
                    "GradCAMPlusPlus": GradCAMPlusPlus,
                    "AblationCAM": AblationCAM,
                    "XGradCAM": XGradCAM,
                    "EigenCAM": EigenCAM,
                    "FullGrad": FullGrad
                }
                cam_class = cam_methods.get(explainer_type, GradCAM)
                
                cam = cam_class(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=img_tensor, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                
                # Save raw grayscale CAM
                plt.imsave(define_save_filename(weights_filename, explainer_type.lower(), image_key), grayscale_cam, cmap='gray')
                return


            def compute_ssims(
                round: int = 1,
                model_weights_folder = "models/0-NoPatterns/shufflenet_v2_x0_5/model_weights/",
                model_name: str = "shufflenet_v2_x0_5",
                explainer_type: str = "GradCAM",
            ):
                # Initialize model and target layers based on model_name
                name_clean = model_name.lower().replace("-", "_").replace(" ", "_")
                if name_clean == "squeezenet1_1":
                    model = models.squeezenet1_1(weights=None, num_classes=10)
                    target_layers = [model.features[12].expand3x3]
                elif name_clean == "shufflenet_v2_x0_5":
                    model = models.shufflenet_v2_x0_5(weights=None, num_classes=10)
                    target_layers = [model.conv5[0]]
                elif "cnn" in name_clean:
                    model = NetA()
                    if hasattr(model, "conv2"):
                        target_layers = [model.conv2]
                    elif hasattr(model, "features"):
                        # In some versions it's a Sequential, conv2 is typically the second Conv2d
                        convs = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
                        if len(convs) >= 2:
                            target_layers = [convs[1]]
                        elif len(convs) == 1:
                            target_layers = [convs[0]]
                        else:
                            log(INFO, f"No conv layers found in model.features for {model_name}")
                            return [], []
                    else:
                        log(INFO, f"Could not determine target layers for CNN model {model_name}")
                        return [], []
                else:
                    log(INFO, f"Model {model_name} not supported for SSIM selection.")
                    return [], []

                # Extract a fresh reference set in-memory at every round (no on-disk cache).
                log(INFO, f"Extracting fresh in-memory reference images for dataset: {DATASET_NAME}")
                dummy_cfg = GLOBAL_CLIENT_DETAILS[0].copy()
                _, testloader = load_data_A(dummy_cfg, 1)
                reference_images = []
                count = 0
                for batch_imgs, _ in testloader:
                    for i in range(len(batch_imgs)):
                        img_tensor = batch_imgs[i]
                        ndarr = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        if ndarr.shape[2] == 1: # Grayscale to RGB for consistency
                            ndarr = np.concatenate([ndarr]*3, axis=2)
                        reference_images.append((f"ref_{count}", ndarr))
                        count += 1

                if not reference_images:
                    log(INFO, f"No reference images could be extracted for {DATASET_NAME}. Skipping selection.")
                    return [], []
                log(INFO, f"[Round {round}] Using {len(reference_images)} in-memory reference images for SSIM selection.")

                server_pt = f"{model_weights_folder}/server/MW_round{round}.pt"
                if not os.path.exists(server_pt):
                    log(INFO, f"Server weights not found for round {round}: {server_pt}. Skipping selection.")
                    return [], []

                client_dirs_info = []
                for client_dir in glob.glob(os.path.join(model_weights_folder, "clients", "*")):
                    base_name = os.path.basename(client_dir)
                    match = re.search(r"(\d+)$", base_name)
                    if not match:
                        log(INFO, f"Skipping non-client folder in model_weights: {base_name}")
                        continue
                    client_dirs_info.append((int(match.group(1)), client_dir))
                client_dirs_info.sort(key=lambda item: item[0])

                clients_pt = []
                client_ids = []
                for parsed_cid, client_dir in client_dirs_info:
                    all_models = glob.glob(os.path.join(client_dir, "MW_round*.pt"))
                    valid = []
                    for p in all_models:
                        m = re.search(r"MW_round(\d+)\.pt$", p)
                        if m and int(m.group(1)) <= round:
                            valid.append((int(m.group(1)), p))
                    if valid:
                        latest_path = max(valid, key=lambda x: x[0])[1]
                        clients_pt.append(latest_path)
                        client_ids.append(parsed_cid)
                    else:
                        log(INFO, f"Nessun modello per client {parsed_cid} fino al round {round}.")

                methods_to_run = ["GradCAM", "HiResCAM", "ScoreCAM", "GradCAMPlusPlus", "AblationCAM", "XGradCAM", "EigenCAM", "FullGrad"] if explainer_type == "All" else [explainer_type]
                expected_cam_calls = len(methods_to_run) * len(reference_images) * (1 + len(clients_pt))
                expected_ssim_calls = len(methods_to_run) * len(reference_images) * len(clients_pt)
                log(
                    INFO,
                    f"[Round {round}] SSIM debug: methods={len(methods_to_run)}, "
                    f"images={len(reference_images)}, clients={len(clients_pt)}, "
                    f"expected_cam_calls={expected_cam_calls}, expected_ssim_pairs={expected_ssim_calls}"
                )

                for _, client_dir in client_dirs_info: # Iterate through client directories to create gradcam_images folders
                    for m in methods_to_run:
                        m_lower = m.lower()
                        if not os.path.exists(f"{client_dir}/{m_lower}_images"):
                            os.makedirs(f"{client_dir}/{m_lower}_images")

                # Also for server
                for m in methods_to_run:
                    m_lower = m.lower()
                    if not os.path.exists(f"{os.path.dirname(server_pt)}/{m_lower}_images"):
                        os.makedirs(f"{os.path.dirname(server_pt)}/{m_lower}_images")
                
                all_results = {} # {method: [avg_ssim_client1, avg_ssim_client2, ...]}

                for method in methods_to_run:
                    method_start = time.time()
                    method_valid_ssim_pairs = 0
                    method_skipped_shape = 0
                    method_skipped_small = 0
                    method_skipped_win = 0
                    # Load server model weights
                    name_clean = model_name.lower().replace("-", "_").replace(" ", "_")
                    if name_clean == "squeezenet1_1":
                        model = load_squeezenet_weights(model, server_pt)
                    elif name_clean == "shufflenet_v2_x0_5":
                        model = load_shufflenet_weights(model, server_pt)
                    else:
                        model.load_state_dict(torch.load(server_pt, weights_only=True))

                    # Run CAM for server model
                    for image_key, image_array in reference_images:
                        run_cam(
                            model=model,
                            target_layers=target_layers,
                            weights_filename=server_pt,
                            image_array=image_array,
                            image_key=image_key,
                            explainer_type=method,
                        )

                    method_ssims = []
                    for client_pt in clients_pt:
                        # Load client model weights
                        if name_clean == "squeezenet1_1":
                            model = load_squeezenet_weights(model, client_pt)
                        elif name_clean == "shufflenet_v2_x0_5":
                            model = load_shufflenet_weights(model, client_pt)
                        else:
                            model.load_state_dict(torch.load(client_pt, weights_only=True))

                        # Run CAM for client model
                        for image_key, image_array in reference_images:
                            run_cam(
                                model=model,
                                target_layers=target_layers,
                                weights_filename=client_pt,
                                image_array=image_array,
                                image_key=image_key,
                                explainer_type=method,
                            )
                        
                        client_ssim_vals = []
                        for image_key, _ in reference_images:
                            server_image = img_as_float(get_image(define_save_filename(server_pt, method.lower(), image_key)))
                            client_image = img_as_float(get_image(define_save_filename(client_pt, method.lower(), image_key)))
                            if server_image.shape != client_image.shape:
                                method_skipped_shape += 1
                                continue

                            min_side = min(server_image.shape[0], server_image.shape[1])
                            if min_side < 3:
                                method_skipped_small += 1
                                continue

                            win_size = min(7, min_side)
                            if win_size % 2 == 0:
                                win_size -= 1
                            if win_size < 3:
                                method_skipped_win += 1
                                continue

                            data_range = max(np.ptp(server_image), np.ptp(client_image))
                            if data_range <= 0:
                                data_range = 1.0

                            channel_axis = -1 if server_image.ndim == 3 else None
                            ssim_val = ssim(
                                server_image,
                                client_image,
                                data_range=data_range,
                                win_size=win_size,
                                channel_axis=channel_axis,
                            )
                            client_ssim_vals.append(ssim_val)
                            method_valid_ssim_pairs += 1
                        if client_ssim_vals:
                            method_ssims.append(float(np.mean(client_ssim_vals)))
                        else:
                            log(INFO, f"No valid SSIM pairs for client model {client_pt}, setting SSIM to 0.0")
                            method_ssims.append(0.0)
                    
                    all_results[method] = method_ssims
                    method_elapsed = time.time() - method_start
                    log(
                        INFO,
                        f"[Round {round}] SSIM method={method}: elapsed={method_elapsed:.3f}s, "
                        f"valid_pairs={method_valid_ssim_pairs}, "
                        f"skipped_shape={method_skipped_shape}, "
                        f"skipped_small={method_skipped_small}, skipped_win={method_skipped_win}"
                    )

                if explainer_type == "All":
                    for i, cid in enumerate(client_ids):
                        metrics_str = ", ".join([f"{m}: {all_results[m][i]:.4f}" for m in methods_to_run])
                        log(INFO, f"[Round {round}] Client {cid} SSIM Metrics -> {metrics_str}")
                
                # Always return GradCAM results for selection if "All" is selected, otherwise the specific method
                return client_ids, all_results.get("GradCAM" if explainer_type == "All" else explainer_type)

            ssim_start = time.time()
            client_ids, ssim_values = compute_ssims(currentRnd, "model_weights", MODEL_NAME, explainer_type)
            ssim_elapsed = time.time() - ssim_start
            ssim_overhead_by_round[currentRnd] = float(ssim_elapsed)
            log(
                INFO,
                f"[Round {currentRnd}] SSIM computation completed in {ssim_elapsed:.2f} seconds "
                f"(clients_scored={len(ssim_values) if ssim_values else 0})"
            )

            if not ssim_values:
                log(INFO, "No SSIM values calculated. Skipping client exclusion.")
                # We need to ensure we return aggregated parameters anyway, but that's handled later.
                # Just skip the exclusion logic.
            else:
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
                excludingCID = client_ids[exclude_idx]
                self.excluded_cid = excludingCID

                log(
                    INFO,
                    f"\nRound {currentRnd} – {values_str}. "
                    f"\nExcluding Client {excludingCID} with SSIM={excluded_val:.4f}"
                )

        model_under_training = GLOBAL_CLIENT_DETAILS[0]["model"]
        if model_under_training not in metrics_history:
            metrics_history[model_under_training] = {key: [global_metrics[model_type][key][-1]]
                                                     for key in global_metrics[model_type]}
        else:
            for key in global_metrics[model_type]:
                metrics_history[model_under_training][key].append(global_metrics[model_type][key][-1])

        metrics_aggregated: Dict[str, Scalar] = {}
        if any(global_metrics.get(model_type, {}).values()):
            metrics_aggregated[model_type] = {
                key: global_metrics[model_type][key][-1]
                if global_metrics[model_type][key] else None
                for key in global_metrics[model_type]
            }

        round_client_ids = [str(client_proxy.cid) for client_proxy, _ in results]
        if hasattr(self.adapt_mgr, "set_last_round_client_ids"):
            self.adapt_mgr.set_last_round_client_ids(round_client_ids)
        next_round_config = self.adapt_mgr.config_next_round(metrics_history, round_total_time)
        agent_time = getattr(self.adapt_mgr, 'adaptation_time', None)
        preprocess_csv(agent_time)
        round_csv = os.path.join(
            performance_dir,
            f"FLwithAP_performance_metrics_round{currentRnd}.csv"
        )
        shutil.copy(csv_file, round_csv)
        if currentRnd >= num_rounds:
            append_ml_experiment_row(build_ml_experiment_row(csv_file))

        log(INFO, metrics_history)
        config_patterns(next_round_config)

        return self.parameters_a, metrics_aggregated

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
        initial_parameters_a=parametersA,
    )
    server_config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    app.run(server_address="[::]:8080", config=ServerConfig(num_rounds=num_rounds))
