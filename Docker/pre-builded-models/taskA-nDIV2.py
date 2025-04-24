from collections import OrderedDict
from logging import INFO
from collections import Counter
import time  
import os
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, KMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1 
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#AP
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False

#CLIENT
global DATASET_TYPE, DATASET_NAME
DATASET_TYPE = ""
DATASET_NAME = "CIFAR10"  # default

# Dataset configurations
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
    }
}

current_dir = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(current_dir, '..', 'configuration') 
config_file = os.path.join(config_dir, 'config.json')

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

class Net(nn.Module):   
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 5)
        self.fc1 = nn.Linear(8 * 5 * 5, 60)
        self.fc2 = nn.Linear(60, 42)
        self.fc3 = nn.Linear(42, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def load_data():
    global DATASET_NAME
    
    if HETEROGENEOUS_DATA_HANDLER:
        with open(config_file, 'r') as f:
            configJSON = json.load(f)
            DATASET_TYPE = configJSON["client_details"][0]["data_distribution_type"]
            DATASET_NAME = configJSON.get("dataset", "CIFAR10") 

    dataset_config = AVAILABLE_DATASETS.get(DATASET_NAME, AVAILABLE_DATASETS["CIFAR10"])
    dataset_class = dataset_config["class"]
    normalize_params = dataset_config["normalize"]

    if HETEROGENEOUS_DATA_HANDLER:
        if DATASET_TYPE == "Random":
           DATASET_TYPE = random.choice(["iid", "non-iid"])

        if DATASET_TYPE == "IID":
            trf = Compose([ToTensor(), Normalize(*normalize_params)])
            trainset = dataset_class("./data", train=True, download=True, transform=trf)
            testset = dataset_class("./data", train=False, download=True, transform=trf)
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

            return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

        if DATASET_TYPE == "non-IID":
            num_non_iid_clients = sum(1 for client in configJSON["client_details"] if client["data_distribution_type"] == "non-IID")
            print(f"Creating {num_non_iid_clients} non-IID clients...")
            samples_per_client = 25000
            alpha = 0.5
            target_samples_per_class = 2500

            trf = Compose([ToTensor(), Normalize(*normalize_params)])
            trainset = dataset_class("./data", train=True, download=True, transform=trf)
            testset = dataset_class("./data", train=False, download=True, transform=trf)

            class_to_indices = {i: [] for i in range(dataset_config["num_classes"])}
            for idx, (_, label) in enumerate(trainset):
                class_to_indices[label].append(idx)
            
            clients_data = []
            
            for client_id in range(num_non_iid_clients):
                proportions = np.random.dirichlet([alpha] * dataset_config["num_classes"])
                class_counts = (proportions * samples_per_client).astype(int)
                
                discrepancy = samples_per_client - class_counts.sum()
                if discrepancy > 0:
                    class_counts[np.argmax(proportions)] += discrepancy
                elif discrepancy < 0:
                    discrepancy = abs(discrepancy)
                    class_counts[np.argmax(proportions)] -= min(discrepancy, class_counts[np.argmax(proportions)])
                
                selected_indices = []
                for cls, count in enumerate(class_counts):
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

            trainloader = DataLoader(subset_augmented, batch_size=32, shuffle=True)
            testloader = DataLoader(testset, batch_size=32, shuffle=False)

            return trainloader, testloader
    else:
        trf = Compose([ToTensor(), Normalize(*normalize_params)])
        trainset = dataset_class("./data", train=True, download=True, transform=trf)
        testset = dataset_class("./data", train=False, download=True, transform=trf)
        return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
    
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
    
def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")

    start_time = time.time()

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
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
    mae_value = None
    try:
        mae_value = mae_score_torch(all_labels.float(), all_preds.float())
    except:
        mae_value = None
    return loss, accuracy, f1, mae_value

def mae_score_torch(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

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
        f1 = f1_per_class.mean().item()
    elif average == 'micro':
        TP = torch.diag(confusion_matrix).sum()
        FP = confusion_matrix.sum() - torch.diag(confusion_matrix).sum()
        FN = FP
        precision_micro = TP / (TP + FP + 1e-8)
        recall_micro = TP / (TP + FN + 1e-8)
        f1 = (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)).item()
    else:
        raise ValueError("Average must be 'macro' or 'micro'")

    return f1

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
