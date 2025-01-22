from flwr.client import ClientApp, NumPyClient
from typing import Dict
from flwr.common.logger import log
from logging import INFO
import time
from datetime import datetime
import os
import psutil
import torch
from taskB import (
    DEVICE as DEVICE_B,
    Net as NetB,
    get_weights as get_weights_B,
    load_data as load_data_B,
    set_weights as set_weights_B,
    train as train_B,
    test as test_B
)

CLIENT_ID = os.getenv("HOSTNAME")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    def __init__(self, cid: str, model_type):
        self.cid = cid
        self.model_type = "taskB"
        self.net = NetB().to(DEVICE_B)       
        self.trainloader, self.testloader = load_data_B()
        self.device = DEVICE_B

    def fit(self, parameters, config):
        # print(f"CLIENT {self.cid} Successfully Configured. Target Model: {self.model_type}", flush=True)
        cpu_start = psutil.cpu_percent(interval=None)

        set_weights_B(self.net, parameters)
        results, training_time = train_B(self.net, self.trainloader, self.testloader, epochs=1, device=self.device)
        communication_start_time = time.time()

        new_parameters = get_weights_B(self.net)

        cpu_end = psutil.cpu_percent(interval=None)
        cpu_usage = (cpu_start + cpu_end) / 2

        metrics = {
            "train_loss": results["train_loss"],
            "train_accuracy": results["train_accuracy"],
            "train_f1": results["train_f1"],
            "val_loss": results["val_loss"],
            "val_accuracy": results["val_accuracy"],
            "val_f1": results["val_f1"],
            "training_time": training_time,
            "cpu_usage": cpu_usage,
            "client_id": self.cid,
            "model_type": self.model_type,
            "communication_start_time": communication_start_time,
        }

        return new_parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print(f"CLIENT {self.cid} ({self.model_type}): Starting evaluation.", flush=True)
        set_weights_B(self.net, parameters)
        loss, accuracy, f1_score = test_B(self.net, self.testloader, self.device)

        print(f"CLIENT {self.cid} ({self.model_type}): Evaluation completed", flush=True)
        metrics = {
            "accuracy": accuracy,
            "client_id": self.cid,
            "model_type": self.model_type,
        }
        return loss, len(self.testloader.dataset), metrics

if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="server:8080",
        client=FlowerClient(cid=CLIENT_ID, model_type="taskB").to_client(),
    )
