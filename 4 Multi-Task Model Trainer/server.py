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
import time
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from APClient import ClientRegistry
import psutil

client_registry = ClientRegistry()

global_metrics = {
    "taskA": {"train_loss": [], "train_accuracy": [], "train_f1": [],"val_loss": [], "val_accuracy": [], "val_f1": []},
    "taskB": {"train_loss": [], "train_accuracy": [], "train_f1": [],"val_loss": [], "val_accuracy": [], "val_f1": []},
}

# Set the non-interactive backend of matplotlib
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
    val_loss = round(global_metrics[model_type]["val_loss"][-1]) if global_metrics[model_type]["val_loss"] else 'N/A'
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
    df['Communication Time'] = pd.to_numeric(df['Communication Time'], errors='coerce')
    max_comm_time_b = df[df['Task'] == 'taskB']['Communication Time'].max()
    print(f"Tempo di comunicazione massimo tra i client B: {max_comm_time_b}")

    if pd.isna(max_comm_time_b):
        print("Non sono stati trovati tempi di comunicazione per i client B. Nessuna correzione effettuata per i tempi di comunicazione.")
    else:
        df.loc[df['Task'] == 'taskA', 'Communication Time'] = df.loc[df['Task'] == 'taskA', 'Communication Time'] - max_comm_time_b
        df['Communication Time'] = df['Communication Time'].clip(lower=0)
        df.loc[df['Task'] == 'taskA', 'Total Client Time'] = df.loc[df['Task'] == 'taskA', 'Training Time'] + df.loc[df['Task'] == 'taskA', 'Communication Time']

    # Map client IDs per task
    unique_tasks = df['Task'].unique()
    client_mappings = {}
    for task in unique_tasks:
        task_clients = sorted(df[df['Task'] == task]['Client ID'].unique())
        for i, old_id in enumerate(task_clients):
            client_number = i + 1
            client_id_new = f'Client {client_number} - {task[-1].upper()}'
            client_mappings[old_id] = client_id_new

    df['Client ID'] = df['Client ID'].map(client_mappings)

    # Extract client number for sorting
    df['Client Number'] = df['Client ID'].str.extract(r'Client (\d+)').astype(int)

    # Ensure 'Task' is ordered with 'taskA' before 'taskB'
    task_order = ['taskA', 'taskB']
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)

    # Sort the DataFrame
    df.sort_values(by=['FL Round', 'Task', 'Client Number'], inplace=True)
    df.drop(columns=['Client Number'], inplace=True)  # Remove helper column

    df.to_csv(csv_file, index=False)

# Update graph generation functions to accommodate new Client IDs
def generate_performance_graphs():
    sns.set_theme(style="ticks")

    df = pd.read_csv(csv_file)

    # No need to remap client IDs here as it's already done
    df = df.reset_index(drop=True)
    df[['Training Time', 'Communication Time', 'Total Client Time']] = df[['Training Time', 'Communication Time', 'Total Client Time']].round(2)

    plt.figure(figsize=(12, 6))
    df_melted = df.melt(id_vars=["Client ID"], value_vars=["Training Time", "Communication Time", "Total Client Time"],
                        var_name="Metric", value_name="Time (seconds)")
    sns.barplot(x="Metric", y="Time (seconds)", hue="Client ID", data=df_melted)
    plt.title('Performance Metrics per Client', fontweight='bold')
    plt.ylabel('Time (seconds)', fontweight='bold')
    plt.legend(title='Client ID', title_fontsize='13', fontsize='10', loc='best', frameon=True)
    plt.tight_layout()

    graph_path = os.path.join(performance_dir, 'performance_metrics.pdf')
    print(f"Saving the graph to: {graph_path}")  
    plt.savefig(graph_path, format="pdf")
    plt.close()

def generate_total_time_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='FL Round', y='Total Client Time', hue='Client ID', data=df, marker="o", markersize=8)
    plt.title('Total Client Time per Round', fontweight='bold')
    plt.ylabel('Total Time (seconds)', fontweight='bold')
    plt.xlabel('FL Round', fontweight='bold')
    min_round = df['FL Round'].min()
    max_round = df['FL Round'].max()
    plt.xticks(range(min_round, max_round + 1))
    plt.tight_layout()

    line_graph_path = os.path.join(performance_dir, 'totalTime_round.pdf')
    print(f"Saving the graph to: {line_graph_path}")
    plt.savefig(line_graph_path, format="pdf")
    plt.close()

# Function to generate CPU usage graph
def generate_cpu_usage_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Client ID", y="CPU Usage (%)", data=df)
    plt.title('CPU Usage per Client', fontweight='bold')
    plt.ylabel('CPU Usage (%)', fontweight='bold')
    plt.xlabel('Client ID', fontweight='bold')
    plt.tight_layout()

    cpu_graph_path = os.path.join(performance_dir, 'cpu_usage_per_client.pdf')
    print(f"Saving the graph to: {cpu_graph_path}")
    plt.savefig(cpu_graph_path, format="pdf")
    
    if os.path.exists(cpu_graph_path):
        print(f"The graph was successfully created in {cpu_graph_path}")
    else:
        print(f"Error: the file {cpu_graph_path} was not created.")
        
    plt.close()

# Function to generate training time graph
def generate_training_time_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='FL Round', y='Training Time', hue='Client ID', data=df, marker="o", markersize=8)
    plt.title('Training Time per Round per Client', fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontweight='bold')
    plt.xlabel('FL Round', fontweight='bold')
    min_round = df['FL Round'].min()
    max_round = df['FL Round'].max()
    plt.xticks(range(min_round, max_round + 1))
    plt.tight_layout()

    line_graph_path = os.path.join(performance_dir, 'trainingTime_round.pdf')
    print(f"Saving the graph to: {line_graph_path}")
    plt.savefig(line_graph_path, format="pdf")
    
    if os.path.exists(line_graph_path):
        print(f"The graph was successfully created in {line_graph_path}")
    else:
        print(f"Error: the file {line_graph_path} was not created.")
        
    plt.close()

# Function to generate communication time graph
def generate_communication_time_graph():
    sns.set_theme(style="ticks")
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='FL Round', y='Communication Time', hue='Client ID', data=df, marker="o", markersize=8)
    plt.title('Communication Time per Round per Client', fontweight='bold')
    plt.ylabel('Communication Time (seconds)', fontweight='bold')
    plt.xlabel('FL Round', fontweight='bold')
    min_round = df['FL Round'].min()
    max_round = df['FL Round'].max()
    plt.xticks(range(min_round, max_round + 1))

    plt.tight_layout()

    line_graph_path = os.path.join(performance_dir, 'communicationTime_round.pdf')
    print(f"Saving the graph to: {line_graph_path}")
    plt.savefig(line_graph_path, format="pdf")
    
    if os.path.exists(line_graph_path):
        print(f"The graph was successfully created in {line_graph_path}")
    else:
        print(f"Error: the file {line_graph_path} was not created.")
        
    plt.close()

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
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_f1s = [num_examples * m["val_f1"] for num_examples, m in metrics]

    avg_train_loss = sum(train_losses) / total_examples
    avg_train_accuracy = sum(train_accuracies) / total_examples
    avg_train_f1 = sum(train_f1s) / total_examples
    avg_val_loss = sum(val_losses) / total_examples
    avg_val_accuracy = sum(val_accuracies) / total_examples
    avg_val_f1 = sum(val_f1s) / total_examples

    global_metrics[task_type]["train_loss"].append(avg_train_loss)
    global_metrics[task_type]["train_accuracy"].append(avg_train_accuracy)
    global_metrics[task_type]["train_f1"].append(avg_train_f1)
    global_metrics[task_type]["val_loss"].append(avg_val_loss)
    global_metrics[task_type]["val_accuracy"].append(avg_val_accuracy)
    global_metrics[task_type]["val_f1"].append(avg_val_f1)

    # Collect client data for this task
    client_data_list = []
    for num_examples, m in metrics:
        client_id = m.get("client_id")
        model_type = m.get("model_type")
        training_time = m.get("training_time")
        cpu_usage = m.get("cpu_usage")
        communication_start_time = m.get("communication_start_time")
       
        if client_id:
            if not client_registry.is_registered(client_id):
                client_registry.register_client(client_id, model_type)
          
            srt2 = time_between_rounds
            communication_time = srt2 - training_time
            total_time = training_time + communication_time
            client_data_list.append((client_id, training_time, communication_time, total_time, cpu_usage, model_type, srt1, srt2))

    # Now write the client data, setting already_logged = True for all except the last client
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
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
        "val_f1": avg_val_f1,
    }

# Initialize weights separately for taskA and taskB
parametersA = ndarrays_to_parameters(get_weights_A(NetA()))
parametersB = ndarrays_to_parameters(get_weights_B(NetB()))

def print_results():
    # Collect clients for each task using cid
    clients_taskA = [cid for cid, model in client_model_mapping.items() if model == "taskA" and len(cid) <= 12]
    clients_taskB = [cid for cid, model in client_model_mapping.items() if model == "taskB" and len(cid) <= 12]

    print("\nResults for Model A:")
    print(f"  Clients: {clients_taskA}")
    print(f"  Train loss: {global_metrics['taskA']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskA']['train_accuracy']}")
    print(f"  Train F1: {global_metrics['taskA']['train_f1']}")
    print(f"  Val loss: {global_metrics['taskA']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskA']['val_accuracy']}")
    print(f"  Val F1: {global_metrics['taskA']['val_f1']}")

    print("\nResults for Model B:")
    print(f"  Clients: {clients_taskB}")
    print(f"  Train loss: {global_metrics['taskB']['train_loss']}")
    print(f"  Train accuracy: {global_metrics['taskB']['train_accuracy']}")
    print(f"  Train F1: {global_metrics['taskB']['train_f1']}")
    print(f"  Val loss: {global_metrics['taskB']['val_loss']}")
    print(f"  Val accuracy: {global_metrics['taskB']['val_accuracy']}")
    print(f"  Val F1: {global_metrics['taskB']['val_f1']}\n")

# Initialize the client_model_mapping dictionary
client_model_mapping = {}

previous_round_end_time = time.time() 

# Definition of the custom strategy
class MultiModelStrategy(Strategy):
    def __init__(self, initial_parameters_a: Parameters, initial_parameters_b: Parameters):
        self.parameters_a = initial_parameters_a
        self.parameters_b = initial_parameters_b

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        # Return None because we use separate initial parameters for A and B
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        min_clients = 4

        client_manager.wait_for(min_clients) 

        # Sample available clients after reaching the minimum number
        clients = client_manager.sample(num_clients=min_clients)

        fit_configurations = []

        task_flag = True  # Start with True for taskA

        for i, client in enumerate(clients):
            client_id = client.cid

            # Alternate between taskA and taskB
            if task_flag:
                fit_ins = FitIns(self.parameters_a, {})
                model_type = "taskA"
            else:
                fit_ins = FitIns(self.parameters_b, {})
                model_type = "taskB"
            
            # Alternate the flag
            task_flag = not task_flag
            
            # Map the model to the client
            client_model_mapping[client_id] = model_type

            # Add the configuration
            fit_configurations.append((client, fit_ins))
        
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        from logging import INFO
        
        global previous_round_end_time
        aggregation_start_time = time.time()

        if previous_round_end_time is not None:
            if server_round-1 == 1:
                time_between_rounds = aggregation_start_time - previous_round_end_time
                log(INFO, f"Results Aggregated in {time_between_rounds:.2f} seconds.")
            else:
                time_between_rounds = aggregation_start_time - previous_round_end_time
                log(INFO, f"Results Aggregated in {time_between_rounds:.2f} seconds")
     
        results_a = []
        results_b = []
        training_times = []
        global currentRnd
        currentRnd += 1
        srt1 = 'N/A'
        srt2 = 'N/A'
        
        for client_proxy, fit_res in results:

            client_id = fit_res.metrics.get("client_id")
            model_type = fit_res.metrics.get("model_type")
            training_time = fit_res.metrics.get("training_time")
            communication_start_time = fit_res.metrics.get("communication_start_time") 

            client_model_mapping[client_id] = model_type

            if training_time is not None:
                training_times.append(training_time)              

            if model_type == "taskA":
                results_a.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
            elif model_type == "taskB":
                results_b.append((fit_res.parameters, fit_res.num_examples, fit_res.metrics))
            else:
                print(f"Unknown Model Type for client {client_id}")
                continue

        previous_round_end_time = time.time()

        # Aggrega i parametri per taskA
        if results_a:
            srt1 = max(training_times)
            self.parameters_a = self.aggregate_parameters(results_a, "taskA", srt1, srt2, time_between_rounds)

        # Aggrega i parametri per taskB
        if results_b:
            srt1 = max(training_times)
            self.parameters_b = self.aggregate_parameters(results_b, "taskB", srt1, srt2, time_between_rounds)
 

        metrics_aggregated = {
            "taskA": {
                "train_loss": global_metrics["taskA"]["train_loss"][-1] if global_metrics["taskA"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskA"]["train_accuracy"][-1] if global_metrics["taskA"]["train_accuracy"] else None,
                "train_accuracy": global_metrics["taskA"]["train_accuracy"][-1] if global_metrics["taskA"]["train_accuracy"] else None,
                "val_loss": global_metrics["taskA"]["val_loss"][-1] if global_metrics["taskA"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskA"]["val_accuracy"][-1] if global_metrics["taskA"]["val_accuracy"] else None,
                "val_accuracy": global_metrics["taskA"]["val_accuracy"][-1] if global_metrics["taskA"]["val_accuracy"] else None,
            },
            "taskB": {
                "train_loss": global_metrics["taskB"]["train_loss"][-1] if global_metrics["taskB"]["train_loss"] else None,
                "train_accuracy": global_metrics["taskB"]["train_accuracy"][-1] if global_metrics["taskB"]["train_accuracy"] else None,
                "train_f1": global_metrics["taskA"]["train_f1"][-1] if global_metrics["taskA"]["train_f1"] else None,
                "val_loss": global_metrics["taskB"]["val_loss"][-1] if global_metrics["taskB"]["val_loss"] else None,
                "val_accuracy": global_metrics["taskB"]["val_accuracy"][-1] if global_metrics["taskB"]["val_accuracy"] else None,
                "val_f1": global_metrics["taskA"]["val_f1"][-1] if global_metrics["taskA"]["val_f1"] else None,
            },
        }

        print_results()

        if currentRnd == num_rounds:
            preprocess_csv()
            print("Starting graph generation...")
            generate_performance_graphs()
            generate_cpu_usage_graph()
            generate_total_time_graph()
            generate_training_time_graph()
            generate_communication_time_graph()

        return (self.parameters_a, self.parameters_b), metrics_aggregated

    def aggregate_parameters(self, results, task_type, srt1, srt2,time_between_rounds):
        # Aggregate weights using weighted average based on number of examples
        total_examples = sum([num_examples for _, num_examples, _ in results])
        new_weights = None

        metrics = []
        for client_params, num_examples, client_metrics in results:
            client_weights = parameters_to_ndarrays(client_params)
            weight = num_examples / total_examples
            #print(f"Aggregating parameters for {task_type}, num_examples: {num_examples}, weight: {weight}")
            if new_weights is None:
                new_weights = [w * weight for w in client_weights]
            else:
                new_weights = [nw + w * weight for nw, w in zip(new_weights, client_weights)]
            metrics.append((num_examples, client_metrics))

        # Aggregate metrics
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
    
    strategy = MultiModelStrategy(
        initial_parameters_a=parametersA,  
        initial_parameters_b=parametersB,  
    )

    start_server(
        server_address="[::]:8080",  
        config=ServerConfig(num_rounds=num_rounds),  
        strategy=strategy, 
    )