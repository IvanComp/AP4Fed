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
from rich.panel import Panel
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO
from taskA import Net as NetA, get_weights as get_weights_A
from taskB import Net as NetB, get_weights as get_weights_B
from rich.console import Console
import time
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import json  

global_metrics = {
    "taskA": {"train_loss": [], "train_accuracy": [], "train_f1": [], "val_loss": [], "val_accuracy": [], "val_f1": []},
    "taskB": {"train_loss": [], "train_accuracy": [], "train_f1": [], "val_loss": [], "val_accuracy": [], "val_f1": []},
}

matplotlib.use('Agg')
current_dir = os.path.abspath(os.path.dirname(__file__))

# Path to the 'configuration' directory
config_dir = os.path.join(current_dir, '..', 'configuration') 
config_file = os.path.join(config_dir, 'config.json')

# Read 'rounds' and 'clients' from config.json
if os.path.exists(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        num_rounds = int(config.get('rounds', 10))  # Use 'rounds' from config.json, default to 10
        client_count = int(config.get('clients', 2))  # Use 'clients' from config.json, default to 2
    except Exception as e:
        log(INFO, f"Error reading config.json: {e}")
        num_rounds = 10  # Default value in case of error
        client_count = 2  # Default value in case of error
else:
    log(INFO, f"config.json not found at {config_file}. Using default num_rounds=10 and client_count=2.")
    num_rounds = 10  # Default value if config.json does not exist
    client_count = 2  # Default value if config.json does not exist

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
        'CPU Number', 'Task', 'Train Loss', 'Train Accuracy', 'Train F1',
        'Val Loss', 'Val Accuracy', 'Val F1', 'Total Time of Training Round', 'Total Time of FL Round'
    ])

def log_round_time(client_id, fl_round, training_time, communication_time, total_time, cpu_usage,
                   model_type, already_logged, srt1, srt2):
    train_loss = round(global_metrics[model_type]["train_loss"][-1], 4) if global_metrics[model_type]["train_loss"] else 'N/A'
    train_accuracy = round(global_metrics[model_type]["train_accuracy"][-1], 2) if global_metrics[model_type]["train_accuracy"] else 'N/A'
    train_f1 = round(global_metrics[model_type]["train_f1"][-1], 2) if global_metrics[model_type]["train_f1"] else 'N/A'
    val_loss = round(global_metrics[model_type]["val_loss"][-1], 4) if global_metrics[model_type]["val_loss"] else 'N/A'
    val_accuracy = round(global_metrics[model_type]["val_accuracy"][-1], 2) if global_metrics[model_type]["val_accuracy"] else 'N/A'
    val_f1 = round(global_metrics[model_type]["val_f1"][-1], 2) if global_metrics[model_type]["val_f1"] else 'N/A'

    if already_logged:
        train_loss = ""
        train_accuracy = ""
        train_f1 = ""
        val_loss = ""
        val_accuracy = ""
        val_f1 = ""
        srt1 = ""
        srt2 = ""

    srt1_rounded = round(srt1, 2) if isinstance(srt1, (int, float)) else srt1
    srt2_rounded = round(srt2, 2) if isinstance(srt2, (int, float)) else srt2

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            client_id, fl_round+1, round(training_time, 2), round(communication_time, 2), round(total_time, 2),
            round(cpu_usage, 2), model_type, train_loss, train_accuracy, train_f1,
            val_loss, val_accuracy, val_f1, srt1_rounded, srt2_rounded
        ])

def preprocess_csv():
    import pandas as pd
    df = pd.read_csv(csv_file)
    df['Training Time'] = pd.to_numeric(df['Training Time'], errors='coerce')
    df['Total Time of FL Round'] = pd.to_numeric(df['Total Time of FL Round'], errors='coerce')
    df['Total Time of FL Round'] = df.groupby('FL Round')['Total Time of FL Round'].transform(
        lambda x: [None] * (len(x) - 1) + [x.iloc[-1]]
    )
    df['Total Client Time'] = df['Training Time'] + df['Communication Time']
    unique_tasks = df['Task'].unique()
    client_mappings = {}
    for task in unique_tasks:
        task_clients = sorted(df[df['Task'] == task]['Client ID'].unique())
        for i, old_id in enumerate(task_clients):
            client_number = i + 1
            client_id_new = f'Client {client_number} - {task.upper()}'
            client_mappings[old_id] = client_id_new

    df['Client ID'] = df['Client ID'].map(client_mappings)
    df['Client Number'] = df['Client ID'].str.extract(r'Client (\d+)').astype(int)
    task_order = ['taskA', 'taskB']
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)
    df.sort_values(by=['FL Round', 'Task', 'Client Number'], inplace=True)
    df.drop(columns=['Client Number'], inplace=True)
    df['Task'] = df['Task'].cat.rename_categories({'taskA': 'CIFAR-10', 'taskB': 'FMNIST'})
    df['Client ID'] = df['Client ID'].str.replace(r' - TASK[A|B]', '', regex=True)
    columns_to_move = [
    "Train Loss", "Train Accuracy", "Train F1",
    "Val Loss", "Val Accuracy", "Val F1", "Total Time of Training Round", "Total Time of FL Round"
    ]
    for round_num, group in df.groupby("FL Round"):
        last_client_index = group.index[-1]
        for col in columns_to_move:
            df.loc[last_client_index, col] = group[col].max()
            df.loc[group.index[:-1], col] = None  # Opzionale: svuota i valori negli altri client

    df.to_csv(csv_file, index=False)
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

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
    train_f1 = [num_examples * m["train_f1"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_f1s = [num_examples * m["val_f1"] for num_examples, m in metrics]

    avg_train_loss = sum(train_losses) / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_train_f1 = sum(train_f1) / total_examples
    avg_val_loss = sum(val_losses) / total_examples
    avg_val_accuracy = sum(val_accuracies) / total_examples
    avg_val_f1 = sum(val_f1s) / total_examples

    global_metrics[task_type]["train_loss"].append(avg_train_loss)
    global_metrics[task_type]["train_accuracy"].append(avg_train_accuracy)
    global_metrics[task_type]["train_f1"].append(avg_train_f1)
    global_metrics[task_type]["val_loss"].append(avg_val_loss)
    global_metrics[task_type]["val_accuracy"].append(avg_val_accuracy)
    global_metrics[task_type]["val_f1"].append(avg_val_f1)

    client_data_list = []
    for num_examples, m in metrics:
        client_id = m.get("client_id")
        model_type = m.get("model_type")
        training_time = m.get("training_time")
        cpu_usage = m.get("cpu_usage")
        start_comm_time = m.get("start_comm_time")
        communication_time = time.time() - start_comm_time       
    
        if client_id:
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
        log_round_time(client_id, currentRnd-1, training_time, communication_time, total_time, cpu_usage, model_type, already_logged, srt1, srt2)

    return {
        "train_loss": avg_train_loss,
        "train_accuracy": avg_train_accuracy,
        "train_f1": avg_train_f1,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
        "val_f1": avg_val_f1,
    }

parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
parametersB = ndarrays_to_parameters(get_weights_B(NetB()))

client_model_mapping = {}
previous_round_end_time = time.time() 

class MultiModelStrategy(Strategy):
    def __init__(self, initial_parameters_a: Parameters, initial_parameters_b: Parameters):
        self.parameters_a = initial_parameters_a
        self.parameters_b = initial_parameters_b
    
    #time.sleep(3)
    log(INFO, f"==========================================")
    log(INFO, f"List of Implemented Architectural Patterns")
    enabled_patterns = [pattern for pattern, enabled in config["patterns"].items() if enabled]

    if enabled_patterns:
        formatted_patterns = "\n".join(
            [f"  {pattern.replace('_', ' ').title()} ✅" for pattern in enabled_patterns]
        )
        log(INFO, f"\n{formatted_patterns}")
    
    log(INFO, f"==========================================")
    #time.sleep(5)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        client_manager.wait_for(client_count) 
        clients = client_manager.sample(num_clients=client_count)

        fit_configurations = []

        for client in clients:

            fit_ins = FitIns(self.parameters_a, {})
            model_type = "taskA"

            client_model_mapping[client.cid] = model_type

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

            client_model_mapping[client_id] = model_type

            if training_time is not None:
                training_times.append(training_time)              

            if model_type == "taskA":
                results_a.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
            elif model_type == "taskB":
                # Gestione separata per taskB se necessaria
                pass
            else:
                continue

        previous_round_end_time = time.time()

        if results_a:
            srt1 = max(training_times)
            self.parameters_a = self.aggregate_parameters(results_a, "taskA", srt1, srt2, time_between_rounds)

        metrics_aggregated = {}

        if any(global_metrics["taskA"].values()): 
            metrics_aggregated["CIFAR-10"] = {
                key: global_metrics["taskA"][key][-1] if global_metrics["taskA"][key] else None
                for key in global_metrics["taskA"]
            }

        if any(global_metrics["taskB"].values()): 
            metrics_aggregated["FMNIST"] = {
                key: global_metrics["taskB"][key][-1] if global_metrics["taskB"][key] else None
                for key in global_metrics["taskB"]
            }

        if metrics_aggregated:
            print(metrics_aggregated)

        if currentRnd == num_rounds:
            preprocess_csv()

        return (self.parameters_a, self.parameters_b), metrics_aggregated

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

def server_fn(context: Context):
    strategy = MultiModelStrategy(
        initial_parameters_a=parametersA,
        initial_parameters_b=parametersB
    )
    server_config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    app.run(server_address="[::]:8080", config=ServerConfig(num_rounds=num_rounds))

# if __name__ == "__main__":
#    start_http_server(8000)    
#    strategy = MultiModelStrategy(
#        initial_parameters_a=parametersA,  
#    )
#
#    start_server(
#        server_address="[::]:8080",  
#        config=ServerConfig(num_rounds=num_rounds),  
#        strategy=strategy, 
#    )