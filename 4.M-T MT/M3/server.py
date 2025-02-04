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
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from taskA import Net as NetA, get_weights as get_weights_A
from taskB import Net as NetB, get_weights as get_weights_B
from flwr.common.logger import log
from logging import INFO
import time
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from APClient import ClientRegistry
import psutil
import docker

client_registry = ClientRegistry()

global_metrics = {
    "taskA": {"train_loss": [], "train_accuracy": [], "train_f1": [], "train_mae": [],"val_loss": [], "val_accuracy": [], "val_f1": [], "val_mae": []},
    "taskB": {"train_loss": [], "train_accuracy": [], "train_f1": [], "train_mae": [], "val_loss": [], "val_accuracy": [], "val_f1": [], "val_mae": []},
}

matplotlib.use('Agg')
current_dir = os.path.abspath(os.path.dirname(__file__))

num_rounds = int(os.getenv("NUM_ROUNDS", 10))
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
        'Client ID', 'FL Round', 'Training Time', 'Communication Time', 'Total Client Time',
        'CPU Usage (%)', 'Task', 'Train Loss', 'Train Accuracy', 'Train F1',
        'Val Loss', 'Val Accuracy', 'Val F1', 'Total Time of Training Round', 'Total Time of FL Round'
    ])

def log_round_time(client_id, fl_round, training_time, communication_time, total_time, cpu_usage,
                   model_type, already_logged, srt1, srt2):
    train_loss = round(global_metrics[model_type]["train_loss"][-1]) if global_metrics[model_type]["train_loss"] else 'N/A'
    train_accuracy = round(global_metrics[model_type]["train_accuracy"][-1], 2) if global_metrics[model_type]["train_accuracy"] else 'N/A'
    train_f1 = round(global_metrics[model_type]["train_f1"][-1], 2) if global_metrics[model_type]["train_f1"] else 'N/A'
    train_mae = round(global_metrics[model_type]["train_mae"][-1], 2) if global_metrics[model_type]["train_mae"] else 'N/A'
    val_loss = round(global_metrics[model_type]["val_loss"][-1]) if global_metrics[model_type]["val_loss"] else 'N/A'
    val_accuracy = round(global_metrics[model_type]["val_accuracy"][-1], 2) if global_metrics[model_type]["val_accuracy"] else 'N/A'
    val_f1 = round(global_metrics[model_type]["val_f1"][-1], 2) if global_metrics[model_type]["val_f1"] else 'N/A'
    val_mae = round(global_metrics[model_type]["val_mae"][-1], 2) if global_metrics[model_type]["val_mae"] else 'N/A'

    if already_logged:
        train_loss = ""
        train_accuracy = ""
        train_mae = ""
        train_f1 = ""
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
            client_id, fl_round+1, round(training_time), round(communication_time, 2), round(total_time),
            round(cpu_usage), model_type, train_loss, train_accuracy, train_f1,
            val_loss, val_accuracy, val_f1, srt1_rounded, srt2_rounded
        ])
    client_registry.update_client(client_id, True)

def preprocess_csv():
    import pandas as pd

    df = pd.read_csv(csv_file)
    df['Training Time'] = pd.to_numeric(df['Training Time'], errors='coerce')
    df['Total Time of FL Round'] = pd.to_numeric(df['Total Time of FL Round'], errors='coerce')
    df['Total Time of FL Round'] = df.groupby('FL Round')['Total Time of FL Round'].transform('max')
    df['Total Client Time'] = df['Training Time'] + df['Communication Time']

    unique_tasks = df['Task'].unique()
    client_mappings = {}
    for task in unique_tasks:
        task_clients = sorted(df[df['Task'] == task]['Client ID'].unique())
        for i, old_id in enumerate(task_clients):
            client_number = i + 1
            client_id_new = f'Client {client_number} - {task[-1].upper()}'
            client_mappings[old_id] = client_id_new

    df['Client ID'] = df['Client ID'].map(client_mappings)
    df['Client Number'] = df['Client ID'].str.extract(r'Client (\d+)').astype(int)
    task_order = ['taskA', 'taskB']
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)
    df.sort_values(by=['FL Round', 'Task', 'Client Number'], inplace=True)
    df.drop(columns=['Client Number'], inplace=True)  
    df.to_csv(csv_file, index=False)

def weighted_average_global(metrics, task_type, srt1, srt2, time_between_rounds):
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    if total_examples == 0:
        return {
            "train_loss": float('inf'),
            "train_accuracy": 0.0,
            "train_f1": 0.0,
            "val_loss": float('inf'),
            "val_accuracy": 0.0,
            "val_f1": 0.0,
        }

    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    train_f1s = [num_examples * m["train_f1"] for num_examples, m in metrics]
    train_mae = [num_examples * m["train_mae"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_f1 = [num_examples * m["val_f1"] for num_examples, m in metrics]
    val_mae = [num_examples * m["val_mae"] for num_examples, m in metrics]

    avg_train_loss = sum(train_losses) / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_train_f1 = sum(train_f1s) / total_examples
    avg_train_mae = sum(train_mae) / total_examples
    avg_val_loss = sum(val_losses) / total_examples
    avg_val_accuracy = sum(val_accuracies) / total_examples
    avg_val_f1 = sum(val_f1) / total_examples
    avg_val_mae = sum(val_mae) / total_examples

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
        training_time = m.get("training_time")
        cpu_usage = m.get("cpu_usage")
        start_comm_time = m.get("start_comm_time")
       
        communication_time = time.time() - start_comm_time       
       
        if client_id:
            if not client_registry.is_registered(client_id):
                client_registry.register_client(client_id, model_type)
          
            srt2 = time_between_rounds           
            total_time = training_time + communication_time
            client_data_list.append((client_id, training_time, communication_time, total_time, cpu_usage, model_type, srt1, srt2))

    num_clients = len(client_data_list)
    for idx, client_data in enumerate(client_data_list):
        client_id, training_time, communication_time, total_time, cpu_usage, model_type, srt1, srt2 = client_data
        if idx == num_clients - 1:
            already_logged = False
        else:
            already_logged = True
        log_round_time(client_id, currentRnd-1, training_time, round(communication_time, 2), total_time, cpu_usage, model_type, already_logged, srt1, srt2)

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

#parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
#parametersB = ndarrays_to_parameters(get_weights_B(NetB()))

def print_results():

    clients_taskA = [cid for cid, model in client_model_mapping.items() if model == "taskA" and len(cid) <= 12]
    clients_taskB = [cid for cid, model in client_model_mapping.items() if model == "taskB" and len(cid) <= 12]

    print(f"\nResults for Model A, round {currentRnd}:")
    print(f"  Clients: {clients_taskA}")
    print(f"  Train loss: {global_metrics['taskA']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskA']['train_accuracy']}")
    print(f"  Train F1: {global_metrics['taskA']['train_f1']}")
    print(f"  Train MAE: {global_metrics['taskA']['train_mae']}")
    print(f"  Val loss: {global_metrics['taskA']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskA']['val_accuracy']}")
    print(f"  Val F1: {global_metrics['taskA']['val_f1']}")
    print(f"  MAE: {global_metrics['taskA']['val_mae']}")

    print(f"\nResults for Model B, round {currentRnd}:")
    print(f"  Clients: {clients_taskB}")
    print(f"  Train loss: {global_metrics['taskB']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskB']['train_accuracy']}")
    print(f"  Train F1: {global_metrics['taskB']['train_f1']}")
    print(f"  Train MAE: {global_metrics['taskB']['train_mae']}")
    print(f"  Val loss: {global_metrics['taskB']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskB']['val_accuracy']}")
    print(f"  Val F1: {global_metrics['taskB']['val_f1']}")
    print(f"  MAE: {global_metrics['taskB']['val_mae']}\n")

client_model_mapping = {}
previous_round_end_time = time.time() 

class MultiModelStrategy(Strategy):
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

        # Ottieni tutti i container con etichetta type=clienta
        client_containers = docker_client.containers.list(
            filters={"label": "type=client"}
        )

        min_clients = len(client_containers)
        client_manager.wait_for(min_clients)
        sampled_clients = client_manager.sample(num_clients=min_clients)
        fit_configurations = []
        config = {"taskA": self.parameters_a, "taskB": self.parameters_b}

        for i, client in enumerate(sampled_clients):
            fit_ins = FitIns(parameters=None, config=config)
            fit_configurations.append((client, fit_ins))
        
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        from logging import INFO
        
        global currentRnd, previous_round_end_time
        aggregation_start_time = time.time()
        time_between_rounds = aggregation_start_time - previous_round_end_time if previous_round_end_time else 0
        previous_round_end_time = time.time()
        currentRnd += 1

        results_a = []
        results_b = []
        training_times = []

        for client_proxy, fit_res in results:
            # fit_res.parameters è ora una lista di due elementi: [parameters_taskA, parameters_taskB]
            paramsA = fit_res.parameters[0]
            paramsB = fit_res.parameters[1]
            num_examples = fit_res.num_examples
            metricsA = fit_res.metrics["taskA"]
            metricsB = fit_res.metrics["taskB"]

            training_times.append(metricsA.get("training_time", 0))
            results_a.append((paramsA, num_examples, metricsA))
            results_b.append((paramsB, num_examples, metricsB))

        # Aggregazione separata per ciascun task
        srt1 = max(training_times) if training_times else 'N/A'
        self.parameters_a = self.aggregate_parameters(results_a, "taskA", srt1, 'N/A', time_between_rounds)
        self.parameters_b = self.aggregate_parameters(results_b, "taskB", srt1, 'N/A', time_between_rounds)

        # Restituisco i nuovi pesi per entrambi i modelli e le metriche aggregate (se necessario)
        metrics_aggregated = {
            "Final Results for Model A": {
                "train_loss": global_metrics["taskA"]["train_loss"][-1] if global_metrics["taskA"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskA"]["train_accuracy"][-1] if global_metrics["taskA"]["train_accuracy"] else None,
                "train_f1": global_metrics["taskA"]["train_f1"][-1] if global_metrics["taskA"]["train_f1"] else None,
                "val_loss": global_metrics["taskA"]["val_loss"][-1] if global_metrics["taskA"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskA"]["val_accuracy"][-1] if global_metrics["taskA"]["val_accuracy"] else None,
                "val_f1": global_metrics["taskA"]["val_f1"][-1] if global_metrics["taskA"]["val_f1"] else None,
            },
            "Final Results for Model B": {
                "train_loss": global_metrics["taskB"]["train_loss"][-1] if global_metrics["taskB"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskB"]["train_accuracy"][-1] if global_metrics["taskB"]["train_accuracy"] else None,
                "train_f1": global_metrics["taskB"]["train_f1"][-1] if global_metrics["taskB"]["train_f1"] else None,
                "val_loss": global_metrics["taskB"]["val_loss"][-1] if global_metrics["taskB"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskB"]["val_accuracy"][-1] if global_metrics["taskB"]["val_accuracy"] else None,
                "val_f1": global_metrics["taskB"]["val_f1"][-1] if global_metrics["taskB"]["val_f1"] else None,
            },
        }

        print_results()

        if currentRnd == num_rounds:
            preprocess_csv()

        return ([self.parameters_a, self.parameters_b], metrics_aggregated)

    def aggregate_parameters(self, results, task_type, srt1, srt2, time_between_rounds):
        total_examples = sum(num_examples for _, num_examples, _ in results)
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
    
    parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
    parametersB = ndarrays_to_parameters(get_weights_B(NetB()))

    strategy = MultiModelStrategy(
        initial_parameters_a=parametersA,  
        initial_parameters_b=parametersB,  
    )

    start_server(
        server_address="[::]:8080",  
        config=ServerConfig(num_rounds=num_rounds),  
        strategy=strategy, 
    )