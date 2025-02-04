# flower_client_unificato.py
from flwr.client import start_client, NumPyClient
from flwr.common import ndarrays_to_parameters
import os
import time
import psutil
import torch

from taskA import (
    DEVICE as DEVICE_A,
    Net as NetA,
    get_weights as get_weights_A,
    set_weights as set_weights_A,
    train as train_A,
    test as test_A,
    load_data as load_data_A,
)
from taskB import (
    DEVICE as DEVICE_B,
    Net as NetB,
    get_weights as get_weights_B,
    set_weights as set_weights_B,
    train as train_B,
    test as test_B,
    load_data as load_data_B,
)

class FlowerClient(NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        # Inizializzazione dei due modelli e dei rispettivi loader
        self.netA = NetA().to(DEVICE_A)
        self.netB = NetB().to(DEVICE_B)
        self.trainloaderA, self.testloaderA = load_data_A()
        self.trainloaderB, self.testloaderB = load_data_B()

    def fit(self, parameters, config):
        """
        Il client riceve in config un dizionario con i pesi per entrambi i task.
        Dopo aver aggiornato entrambi i modelli, esegue il training in sequenza.
        """
        # Recupero dei pesi per il taskA e il taskB dal dizionario di configurazione
        init_params_A = config["taskA"]
        init_params_B = config["taskB"]

        # Aggiornamento dei modelli con i pesi ricevuti
        set_weights_A(self.netA, init_params_A)
        set_weights_B(self.netB, init_params_B)

        cpu_start = psutil.cpu_percent(interval=None)

        # Esecuzione del training per il taskA
        resultsA, training_time_A, start_comm_time_A = train_A(
            self.netA, self.trainloaderA, self.testloaderA, epochs=1, device=DEVICE_A
        )
        new_params_A = get_weights_A(self.netA)

        # Esecuzione del training per il taskB
        resultsB, training_time_B, start_comm_time_B = train_B(
            self.netB, self.trainloaderB, self.testloaderB, epochs=1, device=DEVICE_B
        )
        new_params_B = get_weights_B(self.netB)

        cpu_end = psutil.cpu_percent(interval=None)
        cpu_usage = (cpu_start + cpu_end) / 2

        # Utilizziamo, ad esempio, il massimo tra i due tempi di training come indicatore generale
        training_time = max(training_time_A, training_time_B)

        # Creazione del dizionario di metriche in cui si mantiene la struttura separata per ciascun task
        metrics = {
            "taskA": {
                "train_loss": resultsA["train_loss"],
                "train_accuracy": resultsA["train_accuracy"],
                "train_f1": resultsA["train_f1"],
                "train_mae": resultsA.get("train_mae", None),
                "val_loss": resultsA["val_loss"],
                "val_accuracy": resultsA["val_accuracy"],
                "val_f1": resultsA["val_f1"],
                "val_mae": resultsA.get("val_mae", None),
                "training_time": training_time_A,
                "cpu_usage": cpu_usage,
                "client_id": self.cid,
                "start_comm_time": start_comm_time_A,
            },
            "taskB": {
                "train_loss": resultsB["train_loss"],
                "train_accuracy": resultsB["train_accuracy"],
                "train_f1": resultsB["train_f1"],
                "val_loss": resultsB["val_loss"],
                "val_accuracy": resultsB["val_accuracy"],
                "val_f1": resultsB["val_f1"],
                "training_time": training_time_B,
                "cpu_usage": cpu_usage,
                "client_id": self.cid,
                "start_comm_time": start_comm_time_B,
            },
        }

        num_examples = len(self.trainloaderA.dataset)  # Si assume che il dataset per taskA e taskB abbiano la stessa dimensione

        # Restituisco una lista contenente i nuovi parametri per taskA e taskB, il numero di esempi e le metriche aggregate
        return [ndarrays_to_parameters(new_params_A), ndarrays_to_parameters(new_params_B)], num_examples, metrics

    def evaluate(self, parameters, config):
        """
        Il metodo evaluate può essere implementato per eseguire il test su uno o entrambi i modelli.
        Ad esempio, si può scegliere di valutare il modello del taskA oppure entrambi.
        In questo esempio si esegue il test per entrambi i task e si restituisce un dizionario.
        """
        # Per il taskA
        set_weights_A(self.netA, parameters.get("taskA"))
        lossA, num_examples_A, metricsA = test_A(self.netA, self.testloaderA)

        # Per il taskB
        set_weights_B(self.netB, parameters.get("taskB"))
        lossB, num_examples_B, metricsB = test_B(self.netB, self.testloaderB)

        # Si può scegliere di restituire una struttura dati che riporti entrambe le valutazioni
        eval_metrics = {
            "taskA": {
                "loss": lossA,
                "num_examples": num_examples_A,
                **metricsA
            },
            "taskB": {
                "loss": lossB,
                "num_examples": num_examples_B,
                **metricsB
            }
        }
        # Qui si restituisce None come peso aggregato per la valutazione (oppure una media ponderata se necessaria)
        return None, num_examples_A + num_examples_B, eval_metrics

if __name__ == "__main__":
    CLIENT_ID = os.getenv("HOSTNAME")
    start_client(server_address="server:8080", client=FlowerClient(cid=CLIENT_ID).to_client())
