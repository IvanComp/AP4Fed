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
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, KMNIST, OxfordIIITPet
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
import torchvision.models as models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1

# Flag per i pattern
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False

# Variabili per il dataset
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
        "class": None,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "channels": 3,
        "num_classes": 10  # 10 classi selezionate
    },
    "OXFORDIIITPET": {
        "class": OxfordIIITPet,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "channels": 3,
        "num_classes": 37
    }
}

# Percorso del file di configurazione
current_dir = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(current_dir, '..', 'configuration')
config_file = os.path.join(config_dir, 'config.json')

# Funzione per normalizzare il nome del dataset
def normalize_dataset_name(name: str) -> str:
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
    elif name_clean == "OXFORDIIITPET":
        return "OXFORDIIITPET"
    else:
        return name

# Lettura iniziale del file di configurazione e aggiornamento dei flag
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
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

# Classe dinamica per CNN custom con parametri variabili
class CNN_Dynamic(nn.Module):
    def __init__(
        self, num_classes: int, input_size: int, in_ch: int,
        conv1_out: int, conv2_out: int, fc1_out: int, fc2_out: int
    ) -> None:
        super(CNN_Dynamic, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, conv1_out, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        dummy = torch.zeros(1, in_ch, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flat_size = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Funzione per ottenere dinamicamente la classe dei pesi in base al modello
def get_weight_class_dynamic(model_name: str):
    weight_mapping = {
        "cnn": None,  # architettura custom, non usa pesi pretrained
        "alexnet": "AlexNet_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        # ... altri modelli ...
        "vit_l_32": "ViT_L_32_Weights"
    }
    model_name = model_name.lower()
    weight_class_name = weight_mapping.get(model_name, None)
    if weight_class_name is not None:
        return getattr(models, weight_class_name, None)
    return None

# Funzione per ottenere il modello dinamico in base al config
def get_dynamic_model(num_classes: int, model_name: str = None, pretrained: bool = True) -> nn.Module:
    if model_name is None:
        with open(config_file, 'r') as f:
            configJSON = json.load(f)
        model_name = configJSON["client_details"][0].get("model")
    name = model_name.strip().lower().replace("-", "_").replace(" ", "_")

    # modelli custom: cnn base
    if name == "cnn":
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "KMNIST": 28,
            "ImageNet100": 256, "OXFORDIIITPET": 256
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        print(INFO, f"Custom CNN with {input_size} and channels: {in_ch}")
        return CNN_CIFAR(num_classes, input_size) if in_ch == 3 else CNN_MONO(num_classes, input_size)
    # cnn 16k
    if name in ("cnn_16k", "cnn16k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "KMNIST": 28,
            "ImageNet100": 256, "OXFORDIIITPET": 256
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        print(INFO, f"Custom CNN 16k with {input_size} and channels: {in_ch}")
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=3, conv2_out=8,
            fc1_out=60, fc2_out=42
        )
    # cnn 64k
    if name in ("cnn_64k", "cnn64k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "KMNIST": 28,
            "ImageNet100": 256, "OXFORDIIITPET": 256
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        print(INFO, f"Custom CNN 64k with {input_size} and channels: {in_ch}")
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=6, conv2_out=16,
            fc1_out=120, fc2_out=84
        )
    # cnn 256k
    if name in ("cnn_256k", "cnn256k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "KMNIST": 28,
            "ImageNet100": 256, "OXFORDIIITPET": 256
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        print(INFO, f"Custom CNN 256k with {input_size} and channels: {in_ch}")
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=12, conv2_out=32,
            fc1_out=240, fc2_out=168
        )

    # modelli torchvision
    if not hasattr(models, name):
        raise ValueError(f"Modello '{model_name}' non in torchvision.models")
    constructor = getattr(models, name)

    weight_cls = get_weight_class_dynamic(name)
    if pretrained and weight_cls and hasattr(weight_cls, "DEFAULT"):
        print(INFO, f"DEBUG: Usando pesi pretrained per '{name}'")
        model = constructor(weights=weight_cls.DEFAULT, progress=False)
    else:
        print(INFO, f"DEBUG: Carico '{name}' senza pesi pretrained")
        model = constructor(weights=None, progress=False)

    if hasattr(model, "fc"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        print(INFO, f"Adaptation of '{name}' ({in_f}→{num_classes})")
    elif hasattr(model, "head"):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, num_classes)
        print(INFO, f"Adaptation (Head) of '{name}' ({in_f}→{num_classes})")
    elif hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Sequential):
            for i in reversed(range(len(cls))):
                m = cls[i]
                if isinstance(m, nn.Linear):
                    in_f = m.in_features
                    cls[i] = nn.Linear(in_f, num_classes)
                    print(INFO, f"DEBUG: Adapted the classifier[{i}] of '{name}' ({in_f}→{num_classes})")
                    break
                if isinstance(m, nn.Conv2d):
                    out_ch = m.out_channels
                    cls[i] = nn.Conv2d(m.in_channels, num_classes,
                                       kernel_size=m.kernel_size,
                                       stride=m.stride,
                                       padding=m.padding)
                    print(INFO, f"Adaptation (Conv2) of [{i}] of '{name}' ({out_ch}→{num_classes})")
                    break
            model.classifier = cls
        else:
            in_f = cls.in_features
            model.classifier = nn.Linear(in_f, num_classes)
            print(INFO, f"DEBUG: Adapted classifier of '{name}' ({in_f}→{num_classes})")
    else:
        raise NotImplementedError(f"{name} not Supported!")

    return model

# Funzione Net() di utilità
def Net():
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
    ds = configJSON.get("dataset", None)
    if ds is None:
        ds = configJSON["client_details"][0].get("dataset", None)
    dataset_name = normalize_dataset_name(ds)
    model_name = configJSON["client_details"][0].get("model", None)
    num_classes = AVAILABLE_DATASETS[dataset_name]["num_classes"]
    return get_dynamic_model(num_classes, model_name)

# Funzione per caricare i dati
def load_data(dataset_name=None):
    global DATASET_NAME, DATASET_TYPE
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
        DATASET_TYPE = configJSON["client_details"][0].get("data_distribution_type", "")
        if dataset_name is None:
            dataset_name = configJSON.get("dataset", None)
            if dataset_name is None:
                dataset_name = configJSON["client_details"][0].get("dataset", "CIFAR10")
        dataset_name = normalize_dataset_name(dataset_name)
        DATASET_NAME = dataset_name

    if DATASET_NAME not in AVAILABLE_DATASETS:
        raise ValueError(f"[ERROR] Dataset '{DATASET_NAME}' Not Found in AVAILABLE_DATASETS.")
    dataset_config = AVAILABLE_DATASETS[DATASET_NAME]
    normalize_params = dataset_config["normalize"]

    default_sizes = {
        "CIFAR10": 32,
        "CIFAR100": 32,
        "FashionMNIST": 28,
        "KMNIST": 28,
        "FMNIST": 28,
        "ImageNet100": 256,
        "OXFORDIIITPET": 256
    }
    base_size = default_sizes.get(DATASET_NAME, 32)
    model_name = configJSON["client_details"][0].get("model", "resnet18").lower()
    if model_name in ["alexnet", "vgg11", "vgg13", "vgg16", "vgg19"]:
        target_size = 256
    else:
        target_size = base_size

    if DATASET_NAME == "OXFORDIIITPET":
        transform_list = [Resize((base_size, base_size)), CenterCrop(224), ToTensor(), Normalize(*normalize_params)]
    elif DATASET_NAME == "ImageNet100":
        transform_list = [Resize((target_size, target_size)), CenterCrop(224), ToTensor(), Normalize(*normalize_params)]
    else:
        transform_list = []
        if target_size != base_size:
            transform_list.append(Resize((target_size, target_size)))
        transform_list += [ToTensor(), Normalize(*normalize_params)]
    trf = Compose(transform_list)

    if DATASET_NAME == "ImageNet100":
        batch_size = 64
        from torchvision.datasets import ImageFolder
        train_path = os.path.join("./data", "imagenet100-preprocessed", "train")
        test_path = os.path.join("./data", "imagenet100-preprocessed", "test")
        trainset = ImageFolder(train_path, transform=trf)
        testset = ImageFolder(test_path, transform=trf)
        return DataLoader(trainset, batch_size=batch_size, shuffle=True), DataLoader(testset, batch_size=batch_size)
    else:
        batch_size = 32
        dataset_class = dataset_config["class"]

        def get_datasets(transform):
            if DATASET_NAME == "OXFORDIIITPET":
                trainset = dataset_class("./data", split="trainval", download=True, transform=transform)
                testset = dataset_class("./data", split="test", download=True, transform=transform)
            else:
                trainset = dataset_class("./data", train=True, download=True, transform=transform)
                testset = dataset_class("./data", train=False, download=True, transform=transform)
            return trainset, testset

        if HETEROGENEOUS_DATA_HANDLER:
            if DATASET_TYPE == "Random":
                DATASET_TYPE = random.choice(["iid", "non-iid"])

            if DATASET_TYPE.upper() == "IID":
                trf_iid = Compose([ToTensor(), Normalize(*normalize_params)])
                trainset, testset = get_datasets(trf_iid)
                class_to_indices = {i: [] for i in range(dataset_config["num_classes"])}
                for idx, (_, label) in enumerate(trainset):
                    if len(class_to_indices[label]) < 2500:
                        class_to_indices[label].append(idx)
                selected_indices = []
                for indices in class_to_indices.values():
                    selected_indices.extend(indices)
                subset_train = Subset(trainset, selected_indices)
                subset_labels = [trainset[i][1] for i in selected_indices]
                class_counts = Counter(subset_labels)
                class_names = trainset.classes if hasattr(trainset, 'classes') else [str(i) for i in range(dataset_config["num_classes"])]
                distribution_str = ", ".join([f"{class_names[class_index]}: {count} samples" for class_index, count in class_counts.items()])
                print(f"IID Client - Class Distribution: {distribution_str}")
                return DataLoader(trainset, batch_size=batch_size, shuffle=True), DataLoader(testset)
            if DATASET_TYPE.lower() == "non-iid":
                num_non_iid_clients = sum(1 for client in configJSON["client_details"] if client["data_distribution_type"] == "non-IID")
                print(f"Creating {num_non_iid_clients} non-IID clients...")
                samples_per_client = 25000
                alpha = 0.5
                target_samples_per_class = 2500
                trf_non_iid = Compose([ToTensor(), Normalize(*normalize_params)])
                trainset, testset = get_datasets(trf_non_iid)
                class_to_indices = {i: [] for i in range(dataset_config["num_classes"])}
                for idx, (_, label) in enumerate(trainset):
                    class_to_indices[label].append(idx)
                clients_data = []
                for client_id in range(num_non_iid_clients):
                    proportions = np.random.dirichlet([alpha] * dataset_config["num_classes"])
                    class_counts_arr = (proportions * samples_per_client).astype(int)
                    discrepancy = samples_per_client - class_counts_arr.sum()
                    if discrepancy > 0:
                        class_counts_arr[np.argmax(proportions)] += discrepancy
                    elif discrepancy < 0:
                        discrepancy = abs(discrepancy)
                        class_counts_arr[np.argmax(proportions)] -= min(discrepancy, class_counts_arr[np.argmax(proportions)])
                    selected_indices = []
                    for cls, count in enumerate(class_counts_arr):
                        available_indices = class_to_indices[cls]
                        if count > len(available_indices):
                            selected = available_indices.copy()
                        else:
                            selected = random.sample(available_indices, min(count, len(available_indices)))
                        selected_indices.extend(selected)
                    subset_train = Subset(trainset, selected_indices)
                    subset_labels = [trainset[i][1] for i in selected_indices]
                    class_distribution = Counter(subset_labels)
                    clients_data.append((subset_train, class_distribution))
                    class_names = trainset.classes if hasattr(trainset, 'classes') else [str(i) for i in range(dataset_config["num_classes"])]
                    distribution_str = ", ".join([f"{class_names[cls]}: {class_distribution.get(cls, 0)} samples" for cls in range(dataset_config["num_classes"])])
                    print(f"Client non-IID-{client_id+1} Class Distribution: {distribution_str}")
                augmented_clients = augment_with_gan(clients_data, target_samples_per_class)
                subset_augmented, _ = augmented_clients[0]
                trainloader = DataLoader(subset_augmented, batch_size=batch_size, shuffle=True)
                testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
                return trainloader, testloader
        else:
            trainset, testset = get_datasets(trf)
            return DataLoader(trainset, batch_size=batch_size, shuffle=True), DataLoader(testset, batch_size=batch_size)

# Funzione per augmentazione con GAN
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

# Funzione di training
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

# Funzione di test
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

# Calcolo F1
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

# Funzioni per pesi
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
