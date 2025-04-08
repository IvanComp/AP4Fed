import os
import json
import time
import random
import numpy as np
from collections import OrderedDict, Counter
from logging import INFO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from flwr.common.logger import log
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, KMNIST
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torchvision.models as models

# Impostazioni globali
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Flag per pattern architetturali (letti dal file di configurazione)
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False

# Variabili per il dataset (inizializzate come stringhe vuote: verranno aggiornate dinamicamente dalla configurazione)
global DATASET_TYPE, DATASET_NAME
DATASET_TYPE = ""
DATASET_NAME = ""

# Configurazioni dei dataset
AVAILABLE_DATASETS = {
    "CIFAR10": {
        "class": CIFAR10,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "channels": 3,
        "num_classes": 10
    },
    "CIFAR100": {
        "class": CIFAR100,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "channels": 3,
        "num_classes": 100
    },
    "MNIST": {
        "class": MNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "FashionMNIST": {
        "class": FashionMNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "KMNIST": {
        "class": KMNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "ImageNet100": {
        "class": None,  # Placeholder, da sostituire con la classe effettiva se necessario
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "channels": 3,
        "num_classes": 100
    }
}

# Percorso del file di configurazione
current_dir = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(current_dir, '..', 'configuration')
config_file = os.path.join(config_dir, 'config.json')

# Funzione per normalizzare il nome del dataset
def normalize_dataset_name(name: str) -> str:
    # Rimuovo eventuali trattini e confronto in uppercase
    name_clean = name.replace("-", "").upper()
    if name_clean == "CIFAR10":
        return "CIFAR10"
    elif name_clean == "CIFAR100":
        return "CIFAR100"
    elif name_clean == "IMAGENET100":
        return "ImageNet100"
    elif name_clean in ["MNIST"]:
        return "MNIST"
    elif name_clean in ["FASHIONMNIST", "FMNIST"]:
        return "FashionMNIST"
    elif name_clean == "KMNIST":
        return "KMNIST"
    else:
        # Se non è uno dei casi noti, restituisco il nome originario (o potresti sollevare un'eccezione)
        return name

# Lettura iniziale del file di configurazione e aggiornamento delle variabili globali in modo dinamico
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
    # Aggiorno i flag per i pattern
    for pattern_name, pattern_info in configJSON["patterns"].items():
        if pattern_info["enabled"]:
            if pattern_name == "client_selector":
                CLIENT_SELECTOR = True
            elif pattern_name == "client_cluster":
                CLIENT_CLUSTER = True
            elif pattern_name == "message_compressor":
                MESSAGE_COMPRESSOR = True
            elif pattern_name == "multi-task_model_trainer":
                MULTI_TASK_MODEL_TRAINER = True
            elif pattern_name == "heterogeneous_data_handler":
                HETEROGENEOUS_DATA_HANDLER = True
    # Leggo il dataset dalla chiave "dataset" oppure dal primo elemento di "client_details"
    ds = configJSON.get("dataset")
    if not ds:
        ds = configJSON["client_details"][0].get("dataset", None)
    if ds is None:
        raise ValueError("Il file di configurazione non specifica il dataset né tramite la chiave 'dataset' né in 'client_details'.")
    DATASET_NAME = normalize_dataset_name(ds)
    DATASET_TYPE = configJSON["client_details"][0].get("data_distribution_type", "")

# Definizione della CNN per immagini a colori (3 canali)
class CNN_CIFAR(nn.Module):
    def __init__(self, num_classes: int, input_size: int) -> None:
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=5)
        dummy = torch.zeros(1, 3, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flattened_size = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 60)
        self.fc2 = nn.Linear(60, 42)
        self.fc3 = nn.Linear(42, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Definizione della CNN per immagini in scala di grigi (1 canale)
class CNN_MONO(nn.Module):
    def __init__(self, num_classes: int, input_size: int) -> None:
        super(CNN_MONO, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=5)
        dummy = torch.zeros(1, 1, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flattened_size = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 60)
        self.fc2 = nn.Linear(60, 42)
        self.fc3 = nn.Linear(42, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Funzione per ottenere il modello dinamico
def get_dynamic_model(num_classes: int, model_name: str = None, pretrained: bool = False) -> nn.Module:
    if model_name is None:
        with open(config_file, 'r') as f:
            configJSON = json.load(f)
        model_name = configJSON["client_details"][0].get("model", "resnet18")
    model_name = model_name.lower()
    if model_name == "cnn":
        default_sizes = {
            "CIFAR10": 32,
            "CIFAR100": 32,
            "FashionMNIST": 28,
            "KMNIST": 28,
            "FMNIST": 28,
            "ImageNet100": 224
        }
        input_size = default_sizes.get(DATASET_NAME, 32)
        in_channels = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        if in_channels == 3:
            return CNN_CIFAR(num_classes, input_size)
        elif in_channels == 1:
            return CNN_MONO(num_classes, input_size)
        else:
            raise ValueError(f"Numero di canali non supportato: {in_channels}")
    if not hasattr(models, model_name):
        raise ValueError(f"Il modello {model_name} non è disponibile in torchvision.models")
    model_constructor = getattr(models, model_name)
    if pretrained:
        try:
            weight_enum = getattr(models, f"{model_name.upper()}_Weights")
            model = model_constructor(weights=weight_enum.DEFAULT)
        except AttributeError:
            model = model_constructor(weights=None)
    else:
        model = model_constructor(weights=None)
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name.startswith("alexnet"):
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f"Adattamento dinamico non implementato per il modello {model_name}")
    return model

# Funzione Net() usata per creare il modello
def Net():
    num_classes = AVAILABLE_DATASETS[DATASET_NAME]["num_classes"]
    return get_dynamic_model(num_classes, "cnn")

# Funzione load_data aggiornata: include il resize se il modello richiede dimensioni maggiori
def load_data(dataset_name=None):
    global DATASET_NAME, DATASET_TYPE
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
        DATASET_TYPE = configJSON["client_details"][0]["data_distribution_type"]
        if dataset_name is None:
            dataset_name = configJSON.get("dataset", None)
            if dataset_name is None:
                dataset_name = configJSON["client_details"][0].get("dataset", "CIFAR10")
        # Normalizzo il nome del dataset per la lookup nel dizionario
        dataset_name = normalize_dataset_name(dataset_name)
        DATASET_NAME = dataset_name
    dataset_config = AVAILABLE_DATASETS.get(DATASET_NAME, AVAILABLE_DATASETS["CIFAR10"])
    dataset_class = dataset_config["class"]
    normalize_params = dataset_config["normalize"]
    default_sizes = {
        "CIFAR10": 32,
        "CIFAR100": 32,
        "FashionMNIST": 28,
        "KMNIST": 28,
        "FMNIST": 28,
        "ImageNet100": 224
    }
    base_size = default_sizes.get(DATASET_NAME, 32)
    model_name = configJSON["client_details"][0].get("model", "resnet18").lower()
    if model_name in ["alexnet", "vgg11", "vgg13", "vgg16", "vgg19"]:
        target_size = 224
    else:
        target_size = base_size
    transform_list = []
    if target_size != base_size:
        transform_list.append(Resize((target_size, target_size)))
    transform_list += [ToTensor(), Normalize(*normalize_params)]
    trf = Compose(transform_list)
    if HETEROGENEOUS_DATA_HANDLER:
        # Gestione dei casi non-IID (non implementata qui per brevità)
        pass
    else:
        trainset = dataset_class("./data", train=True, download=True, transform=trf)
        testset = dataset_class("./data", train=False, download=True, transform=trf)
        return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset, batch_size=32)

def augment_with_gan(clients_data, target_samples_per_class=2500):
    augmented_clients_data = []
    for idx, (subset, class_distribution) in enumerate(clients_data):
        new_class_distribution = Counter(class_distribution)
        print("\nApplying GAN Augmentation to non-IID Clients...")
        print(f"\nAugmenting non-IID Client {idx+1}:")
        for cls in range(AVAILABLE_DATASETS[DATASET_NAME]["num_classes"]):
            current_count = new_class_distribution[cls]
            class_name = str(cls)
            if hasattr(subset.dataset, 'classes'):
                class_name = subset.dataset.classes[cls]
            if current_count < target_samples_per_class:
                additional_samples = target_samples_per_class - current_count
                new_class_distribution[cls] += additional_samples
                print(f"  {class_name} - Adding {additional_samples} samples to reach {target_samples_per_class}")
            elif current_count > target_samples_per_class:
                samples_to_remove = current_count - target_samples_per_class
                new_class_distribution[cls] -= samples_to_remove
                print(f"  {class_name} - Removing {samples_to_remove} samples to reach {target_samples_per_class}")
            else:
                print(f"  {class_name} - No augmentation needed (current: {current_count})")
        augmented_clients_data.append((subset, new_class_distribution))
    return augmented_clients_data

def train(net, trainloader, valloader, epochs, DEVICE):
    log(INFO, "Starting training...")
    start_time = time.time()
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")
    start_comm_time = time.time()
    train_loss, train_acc, train_f1, train_mae = test(net, trainloader)
    val_loss, val_acc, val_f1, val_mae = test(net, valloader)
    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "train_mae": train_mae,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_mae": val_mae,
    }
    return results, training_time, start_comm_time

def test(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_preds.append(predicted)
            all_labels.append(labels)
    accuracy = correct / len(testloader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score_torch(all_labels, all_preds, num_classes=AVAILABLE_DATASETS[DATASET_NAME]["num_classes"], average='macro')
    try:
        mae_value = torch.mean(torch.abs(all_labels.float() - all_preds.float())).item()
    except Exception:
        mae_value = None
    return loss, accuracy, f1, mae_value

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t.long(), p.long()] += 1
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_per_class = torch.zeros(num_classes)
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        precision[i] = TP / (TP + FP + 1e-8)
        recall[i] = TP / (TP + FN + 1e-8)
        f1_per_class[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
    if average == 'macro':
        return f1_per_class.mean().item()
    elif average == 'micro':
        TP = torch.diag(confusion_matrix).sum()
        FP = confusion_matrix.sum() - torch.diag(confusion_matrix).sum()
        FN = FP
        precision_micro = TP / (TP + FP + 1e-8)
        recall_micro = TP / (TP + FN + 1e-8)
        return (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)).item()
    else:
        raise ValueError("Average must be 'macro' or 'micro'")

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
