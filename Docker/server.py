import base64
import csv
import json
import os
import pickle
import shutil
import time
import re
import zlib
import glob
import numpy as np
from torchvision import models
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from logging import INFO
from typing import List, Tuple, Dict, Optional
import docker
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    FitIns,
    EvaluateIns,
)
from flwr.server import (
    ServerConfig,
    start_server
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.common.logger import log
from taskA import Net as NetA, get_weights as get_weights_A, set_weights as set_weights_A
docker_client = docker.from_env()
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from adaptation import AdaptationManager

current_dir = os.path.dirname(os.path.abspath(__file__))
folders_to_delete = ["performance", "model_weights", "logs"]

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
    num_rounds = int(config.get('rounds', 10))
    client_count = int(config.get('clients', 2))
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
        'Client ID', 'FL Round',
        'Training Time', 'JSD', 'HDH Time', 'Communication Time', 'Total Time of FL Round',
        '# of CPU', 'CPU Usage (%)', 'RAM Usage (%)',
        'Model Type', 'Data Distr. Type', 'Dataset',
        'Train Loss', 'Train Accuracy', 'Train F1', 'Train MAE',
        'Val Loss', 'Val Accuracy', 'Val F1', 'Val MAE',
        'AP List (client_selector,client_cluster,message_compressor,model_co-versioning_registry,multi-task_model_trainer,heterogeneous_data_handler)'
    ])



def log_round_time(
        client_id, fl_round,
        training_time, jsd, hdh_ms, communication_time, time_between_rounds,
        n_cpu, cpu_percent, ram_percent,
        client_model_type, data_distr, dataset_value,
        already_logged, srt1, srt2, agg_key
):
    try:
        client_id = docker_client.containers.get(client_id).name
    except Exception:
        client_id = client_id

    if client_id.startswith("docker-"):
        client_id = client_id[len("docker-"):]
        client_id = client_id.replace("-", " ").title()

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
            ap_list
        ])


def preprocess_csv():
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
                srt2
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
            srt2
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
            agg_model_type
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

class FedAvg(Strategy):
    def __init__(self, initial_parameters_a: Parameters):
        self.round_start_time: float | None = None
        self.parameters_a = initial_parameters_a
        self._send_ts = {}
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

        ADAPTATION = config.get('adaptation', False).strip()

        if ADAPTATION == "None":
            log(INFO, "Adaptation Disabled ❌")
            self.adapt_mgr = AdaptationManager(False, config)
        else:
            if "Voting" in ADAPTATION:
                ADAPTATION = "Voting-Based"
            elif "Voting" in ADAPTATION:
                ADAPTATION = "Voting-Based"
            elif "Role" in ADAPTATION:
                ADAPTATION = "Role-Based"
            elif "Debate" in ADAPTATION:
                ADAPTATION = "Debate-Based"

            self.adapt_mgr = AdaptationManager(True, config)
            log(INFO, f"Adaptation Enabled ✅ - {ADAPTATION}")

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

        num_fit = available
        clients: List[ClientProxy] = client_manager.sample(
            num_clients=num_fit,
            min_num_clients=1
        )

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
                    with open("exclusion_log.txt", "r") as f:
                        excluded_cid = f.read().strip()
                        log(INFO, f"[Round {currentRnd}] Client {excluded_cid} excluded")

        agg_start = time.time()
        round_total_time = time.time() - self.round_start_time
        log(INFO, f"Results Aggregated in {round_total_time:.2f} seconds.")

        results_a = []
        training_times = []
        currentRnd += 1

        for client_proxy, fit_res in results:
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

            if CLIENT_SELECTOR and selection_strategy == "SSIM-Based" and excluded_cid:
                if str(client_id) == excluded_cid:
                    log(INFO, f"[Round {currentRnd}] Skipping aggregation of client {client_id}")
                    continue

            if MESSAGE_COMPRESSOR and compressed_parameters_b64:
                compressed = base64.b64decode(compressed_parameters_b64)
                decompressed = pickle.loads(zlib.decompress(compressed))
                fit_res.parameters = ndarrays_to_parameters(decompressed)

            if training_time is not None:
                training_times.append(training_time)

            results_a.append((fit_res.parameters, fit_res.num_examples, metrics))

        previous_round_end_time = time.time()
        max_train = max(training_times) if training_times else 0.0
        agg_end = time.time()
        aggregation_time = agg_end - agg_start

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
                #img_explanation = show_cam_on_image(img, grayscale_cam, use_rgb=True)
                #plt.imsave(define_save_filename(weights_filename, "gradcam", image_filename), img_explanation)
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

        preprocess_csv()
        round_csv = os.path.join(
            performance_dir,
            f"FLwithAP_performance_metrics_round{currentRnd}.csv"
        )
        shutil.copy(csv_file, round_csv)
        next_round_config = self.adapt_mgr.config_next_round(metrics_history, round_total_time)
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


if __name__ == "__main__":
    strategy = FedAvg(
        initial_parameters_a=parametersA,
    )

    start_server(
        server_address="[::]:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )