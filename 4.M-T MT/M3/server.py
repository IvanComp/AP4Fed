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
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO
import time
import csv
import os
import psutil
import docker

# Per gestire la registrazione e l’identità dei client
from APClient import ClientRegistry
client_registry = ClientRegistry()

# Import del codice relativo ai due task
from taskA import Net as NetA, get_weights as get_weights_A
from taskB import Net as NetB, get_weights as get_weights_B

global_metrics = {
    "taskA": {"train_loss": [], "train_accuracy": [], "train_f1": [], "train_mae": [], "val_loss": [], "val_accuracy": [], "val_f1": [], "val_mae": []},
    "taskB": {"train_loss": [], "train_accuracy": [], "train_f1": [], "train_mae": [], "val_loss": [], "val_accuracy": [], "val_f1": [], "val_mae": []},
}

performance_dir = './performance/'
if not os.path.exists(performance_dir):
    os.makedirs(performance_dir)

csv_file = os.path.join(performance_dir, 'FLwithAP_performance_metrics.csv')
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Client Time',
        'CPU Usage (%)', 'Task', 'Train Loss', 'Train Accuracy', 'Train F1', 'Train MAE',
        'Val Loss', 'Val Accuracy', 'Val F1', 'Val MAE', 'Total Time of Training Round', 'Total Time of FL Round'
    ])

num_rounds = int(os.getenv("NUM_ROUNDS", 10))
currentRnd = 0
previous_round_end_time = time.time()

def preprocess_csv():
    import pandas as pd

    df = pd.read_csv(csv_file)
    df['Training Time'] = pd.to_numeric(df['Training Time'], errors='coerce')
    df['Total Time of FL Round'] = pd.to_numeric(df['Total Time of FL Round'], errors='coerce')
    df['Total Time of FL Round'] = df.groupby('FL Round')['Total Time of FL Round'].transform('max')
    df['Total Client Time'] = df['Training Time'] + df['Communication Time']

    client_mappings = {}

    for i in range(len(df)):
        old_id = df.at[i, 'Client ID']
        task = df.at[i, 'Task']
        if (old_id, task) not in client_mappings:
            existing_for_task = [k for k in client_mappings if k[1] == task]
            client_number = len(existing_for_task) + 1
            client_mappings[(old_id, task)] = f"Client {client_number} - {task[-1].upper()}"
        df.at[i, 'Client ID'] = client_mappings[(old_id, task)]

    task_order = ['taskA', 'taskB']
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)

    df['Client Number'] = df['Client ID'].str.extract(r'Client (\d+)').astype(int)
    df.sort_values(by=['FL Round', 'Task', 'Client Number'], inplace=True)
    df.drop(columns=['Client Number'], inplace=True)

    df.to_csv(csv_file, index=False)

def log_round_time(client_id, fl_round, training_time, communication_time, total_time, cpu_usage,
                   model_type, already_logged, srt1, srt2):
    train_loss = round(global_metrics[model_type]["train_loss"][-1], 2) if global_metrics[model_type]["train_loss"] else 'N/A'
    train_accuracy = round(global_metrics[model_type]["train_accuracy"][-1], 4) if global_metrics[model_type]["train_accuracy"] else 'N/A'
    train_f1 = round(global_metrics[model_type]["train_f1"][-1], 4) if global_metrics[model_type]["train_f1"] else 'N/A'
    train_mae = round(global_metrics[model_type]["train_mae"][-1], 4) if global_metrics[model_type]["train_mae"] else 'N/A'
    val_loss = round(global_metrics[model_type]["val_loss"][-1], 2) if global_metrics[model_type]["val_loss"] else 'N/A'
    val_accuracy = round(global_metrics[model_type]["val_accuracy"][-1], 4) if global_metrics[model_type]["val_accuracy"] else 'N/A'
    val_f1 = round(global_metrics[model_type]["val_f1"][-1], 4) if global_metrics[model_type]["val_f1"] else 'N/A'
    val_mae = round(global_metrics[model_type]["val_mae"][-1], 4) if global_metrics[model_type]["val_mae"] else 'N/A'

    if already_logged:
        train_loss = ""
        train_accuracy = ""
        train_f1 = ""
        train_mae = ""
        val_loss = ""
        val_accuracy = ""
        val_f1 = ""
        val_mae = ""
        srt1 = ""
        srt2 = ""

    srt1_rounded = round(srt1) if isinstance(srt1, (int, float)) else srt1
    srt2_rounded = round(srt2) if isinstance(srt2, (int, float)) else srt2

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            client_id, fl_round + 1, round(training_time, 2), round(communication_time, 2), round(total_time, 2),
            round(cpu_usage, 2), model_type, train_loss, train_accuracy, train_f1, train_mae,
            val_loss, val_accuracy, val_f1, val_mae, srt1_rounded, srt2_rounded
        ])
    client_registry.update_client(client_id, True)

def weighted_average_global(metrics, task_type, srt1, srt2, time_between_rounds):
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
    train_f1s = [num_examples * m["train_f1"] for num_examples, m in metrics]
    train_maes = [num_examples * m["train_mae"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_f1s = [num_examples * m["val_f1"] for num_examples, m in metrics]
    val_maes = [num_examples * m["val_mae"] for num_examples, m in metrics]

    avg_train_loss = sum(train_losses) / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_train_f1 = sum(train_f1s) / total_examples
    avg_train_mae = sum(train_maes) / total_examples
    avg_val_loss = sum(val_losses) / total_examples
    avg_val_accuracy = sum(val_accuracies) / total_examples
    avg_val_f1 = sum(val_f1s) / total_examples
    avg_val_mae = sum(val_maes) / total_examples

    global_metrics[task_type]["train_loss"].append(avg_train_loss)
    global_metrics[task_type]["train_accuracy"].append(avg_train_accuracy)
    global_metrics[task_type]["train_f1"].append(avg_train_f1)
    global_metrics[task_type]["train_mae"].append(avg_train_mae)
    global_metrics[task_type]["val_loss"].append(avg_val_loss)
    global_metrics[task_type]["val_accuracy"].append(avg_val_accuracy)
    global_metrics[task_type]["val_f1"].append(avg_val_f1)
    global_metrics[task_type]["val_mae"].append(avg_val_mae)

    client_data_list = []
    for num_examples, m in metrics:
        client_id = m.get("client_id")
        model_type = m.get("model_type")
        training_time = m.get("training_time", 0.0)
        cpu_usage = m.get("cpu_usage", 0.0)
        start_comm_time = m.get("start_comm_time", time.time())
        communication_time = time.time() - start_comm_time

        if client_id:
            if not client_registry.is_registered(client_id):
                client_registry.register_client(client_id, model_type)
            total_time = training_time + communication_time
            client_data_list.append((client_id, training_time, communication_time, total_time, cpu_usage, model_type, srt1, srt2))

    num_clients = len(client_data_list)
    for idx, client_data in enumerate(client_data_list):
        client_id, training_time, communication_time, total_time, cpu_usage, model_type, srt1, srt2 = client_data
        if idx == num_clients - 1:
            already_logged = False
        else:
            already_logged = True
        log_round_time(client_id, currentRnd-1, training_time, communication_time, total_time, cpu_usage, model_type, already_logged, srt1, srt2)

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

client_model_mapping = {}

def print_results():
    clients_taskA = [cid for cid, model in client_model_mapping.items() if model == "taskA" and len(cid) <= 12]
    clients_taskB = [cid for cid, model in client_model_mapping.items() if model == "taskB" and len(cid) <= 12]

    print(f"\nResults for Model 1 (Semantic Segmentation), round {currentRnd}:")
    print(f"  Clients: {clients_taskA}")
    print(f"  Train loss: {global_metrics['taskA']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskA']['train_accuracy']}")
    print(f"  Train F1: {global_metrics['taskA']['train_f1']}")
    print(f"  Train MAE: {global_metrics['taskA']['train_mae']}")
    print(f"  Val loss: {global_metrics['taskA']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskA']['val_accuracy']}")
    print(f"  Val F1: {global_metrics['taskA']['val_f1']}")
    print(f"  Val MAE: {global_metrics['taskA']['val_mae']}")

    print(f"\nResults for Model 2 (Depth Estimation), round {currentRnd}:")
    print(f"  Clients: {clients_taskB}")
    print(f"  Train loss: {global_metrics['taskB']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskB']['train_accuracy']}")
    print(f"  Train F1: {global_metrics['taskB']['train_f1']}")
    print(f"  Train MAE: {global_metrics['taskB']['train_mae']}")
    print(f"  Val loss: {global_metrics['taskB']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskB']['val_accuracy']}")
    print(f"  Val F1: {global_metrics['taskB']['val_f1']}")
    print(f"  Val MAE: {global_metrics['taskB']['val_mae']}\n")

parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
parametersB = ndarrays_to_parameters(get_weights_B(NetB()))

class MultiTaskStrategy(Strategy):
    def __init__(self, initial_parameters_a: Parameters, initial_parameters_b: Parameters):
        self.parameters_a = initial_parameters_a
        self.parameters_b = initial_parameters_b

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        docker_client = docker.from_env()
        client_containers = docker_client.containers.list(filters={"label": "type=client"})
        min_clients = len(client_containers)
        client_manager.wait_for(min_clients)
        sampled_clients = client_manager.sample(num_clients=min_clients)

        # Round dispari -> taskA, round pari -> taskB (semplice esempio)
        if server_round % 2 == 1:
            model_type = "taskA"
            selected_params = self.parameters_a
        else:
            model_type = "taskB"
            selected_params = self.parameters_b

        fit_configurations = []
        for client in sampled_clients:
            client_id = client.cid
            fit_ins = FitIns(selected_params, {"model_type": model_type})
            client_model_mapping[client_id] = model_type
            fit_configurations.append((client, fit_ins))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        global previous_round_end_time
        global currentRnd
        aggregation_start_time = time.time()

        if previous_round_end_time is not None:
            time_between_rounds = aggregation_start_time - previous_round_end_time
            log(INFO, f"Results Aggregated in {time_between_rounds:.2f} seconds.")
        else:
            time_between_rounds = 0.0

        currentRnd = (server_round + 1) // 2
        if currentRnd == num_rounds:
            preprocess_csv()

        results_a = []
        results_b = []
        training_times = []
        srt1 = 'N/A'
        srt2 = 'N/A'

        # Leggiamo tutti i FitRes e li smistiamo in base al task
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id")
            model_type = fit_res.metrics.get("model_type")
            training_time = fit_res.metrics.get("training_time", 0.0)
            if training_time:
                training_times.append(training_time)

            if model_type == "taskA":
                results_a.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
            elif model_type == "taskB":
                results_b.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))

        previous_round_end_time = time.time()

        if training_times:
            srt1 = max(training_times)

        # Aggregazione per i due task
        if results_a:
            self.parameters_a = self.aggregate_parameters(results_a, "taskA", srt1, srt2, time_between_rounds)
        if results_b:
            self.parameters_b = self.aggregate_parameters(results_b, "taskB", srt1, srt2, time_between_rounds)

        metrics_aggregated = {
            "Final Results for Model A": {
                "train_loss": global_metrics["taskA"]["train_loss"][-1] if global_metrics["taskA"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskA"]["train_accuracy"][-1] if global_metrics["taskA"]["train_accuracy"] else None,
                "train_f1": global_metrics["taskA"]["train_f1"][-1] if global_metrics["taskA"]["train_f1"] else None,
                "train_mae": global_metrics["taskA"]["train_mae"][-1] if global_metrics["taskA"]["train_mae"] else None,
                "val_loss": global_metrics["taskA"]["val_loss"][-1] if global_metrics["taskA"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskA"]["val_accuracy"][-1] if global_metrics["taskA"]["val_accuracy"] else None,
                "val_f1": global_metrics["taskA"]["val_f1"][-1] if global_metrics["taskA"]["val_f1"] else None,
                "val_mae": global_metrics["taskA"]["val_mae"][-1] if global_metrics["taskA"]["val_mae"] else None,
            },
            "Final Results for Model B": {
                "train_loss": global_metrics["taskB"]["train_loss"][-1] if global_metrics["taskB"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskB"]["train_accuracy"][-1] if global_metrics["taskB"]["train_accuracy"] else None,
                "train_f1": global_metrics["taskB"]["train_f1"][-1] if global_metrics["taskB"]["train_f1"] else None,
                "train_mae": global_metrics["taskB"]["train_mae"][-1] if global_metrics["taskB"]["train_mae"] else None,
                "val_loss": global_metrics["taskB"]["val_loss"][-1] if global_metrics["taskB"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskB"]["val_accuracy"][-1] if global_metrics["taskB"]["val_accuracy"] else None,
                "val_f1": global_metrics["taskB"]["val_f1"][-1] if global_metrics["taskB"]["val_f1"] else None,
                "val_mae": global_metrics["taskB"]["val_mae"][-1] if global_metrics["taskB"]["val_mae"] else None,
            },
        }

        print_results()

        if currentRnd == num_rounds:
            preprocess_csv()

        # Se round dispari -> restituisco parametri di A, altrimenti di B
        if server_round % 2 == 1:
            return self.parameters_a, metrics_aggregated
        else:
            return self.parameters_b, metrics_aggregated

    def aggregate_parameters(self, results, task_type, srt1, srt2, time_between_rounds):
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

        weighted_average_global(metrics, task_type, srt1, srt2, time_between_rounds)
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
    strategy = MultiTaskStrategy(
        initial_parameters_a=parametersA,
        initial_parameters_b=parametersB,
    )
    start_server(
        server_address="[::]:8080",
        config=ServerConfig(num_rounds=num_rounds * 2),
        strategy=strategy,
    )
