import os
import psutil
import torch
from flwr.client import NumPyClient, start_client
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from APClient import ClientRegistry
from taskA import (
    Net as NetA,
    get_weights as get_weights_A,
    set_weights as set_weights_A,
    load_data as load_data_A,
    train as train_A,
    test as test_A
)
from taskB import (
    Net as NetB,
    get_weights as get_weights_B,
    set_weights as set_weights_B,
    load_data as load_data_B,
    train as train_B,
    test as test_B
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENT_ID = os.getenv("HOSTNAME", "default_client_id")
client_registry = ClientRegistry()

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"Using CUDA")
else:
    print(f"Using CPU")

class FlowerClient(NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        # Inizializza un modello (e relativi loader) per Task A
        self.modelA = NetA().to(DEVICE)
        self.trainloaderA, self.testloaderA = load_data_A()
        # Inizializza un modello (e relativi loader) per Task B
        self.modelB = NetB().to(DEVICE)
        self.trainloaderB, self.testloaderB = load_data_B()
        client_registry.register_client(cid, None)

    def fit(self, parameters, config):
        model_type = config.get("model_type", "taskA")
        cpu_start = psutil.cpu_percent(interval=None)

        if model_type == "taskA":
            set_weights_A(self.modelA, parameters)
            results, training_time, start_comm_time = train_A(
                self.modelA,
                self.trainloaderA,
                self.testloaderA,
                epochs=1,
                device=DEVICE
            )
            new_parameters = get_weights_A(self.modelA)
            dataset_size = len(self.trainloaderA.dataset)
        elif model_type == "taskB":
            set_weights_B(self.modelB, parameters)
            results, training_time, start_comm_time = train_B(
                self.modelB,
                self.trainloaderB,
                self.testloaderB,
                epochs=1,
                device=DEVICE
            )
            new_parameters = get_weights_B(self.modelB)
            dataset_size = len(self.trainloaderB.dataset)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        cpu_end = psutil.cpu_percent(interval=None)
        cpu_usage = (cpu_start + cpu_end) / 2

        # Qui usiamo .get("<chiave>", valore_di_default) per evitare errori se la chiave non esiste
        metrics = {
            "train_loss": results.get("train_loss", 0.0),
            "train_accuracy": results.get("train_accuracy", 0.0),
            "train_f1": results.get("train_f1", 0.0),
            "train_mae": results.get("train_mae", 0.0),    
            "val_loss": results.get("val_loss", 0.0),
            "val_accuracy": results.get("val_accuracy", 0.0),
            "val_f1": results.get("val_f1", 0.0),
            "val_mae": results.get("val_mae", 0.0),      
            "training_time": training_time,
            "cpu_usage": cpu_usage,
            "client_id": self.cid,
            "model_type": model_type,
            "start_comm_time": start_comm_time,
        }

        return new_parameters, dataset_size, metrics

    def evaluate(self, parameters, config):
        model_type = config.get("model_type", "taskA")

        if model_type == "taskA":
            set_weights_A(self.modelA, parameters)
            loss, size, eval_metrics = test_A(self.modelA, self.testloaderA)
        elif model_type == "taskB":
            set_weights_B(self.modelB, parameters)
            loss, size, eval_metrics = test_B(self.modelB, self.testloaderB)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        eval_metrics["client_id"] = self.cid
        eval_metrics["model_type"] = model_type
        return loss, size, eval_metrics

if __name__ == "__main__":
    start_client(server_address="server:8080", client=FlowerClient(cid=CLIENT_ID).to_client())