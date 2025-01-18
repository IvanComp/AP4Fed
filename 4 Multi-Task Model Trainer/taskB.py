import json
import time  # Keep time to measure training times
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import os
from flwr.common.logger import log
from logging import INFO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Lambda,
    Resize,
)
import numpy as np
import hashlib

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Normalization constants for Fashion-MNIST
FM_NORMALIZATION = ((0.2860,), (0.3530,))

def expand_channels(image):
    return image.expand(3, -1, -1)

EVAL_TRANSFORMS = Compose([
    Resize(32),
    ToTensor(),
    Lambda(expand_channels),
    Normalize((0.2860,) * 3, (0.3530,) * 3),
])

TRAIN_TRANSFORMS = Compose([
    Resize(32),
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Lambda(expand_channels),
    Normalize((0.2860,) * 3, (0.3530,) * 3),
])

class Net(nn.Module):
    """Model (simple CNN adapted for Fashion-MNIST)"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)

def load_data():
    """Load FashionMNIST (training and test set) with appropriate transformations."""
    train_transform = TRAIN_TRANSFORMS
    test_transform = EVAL_TRANSFORMS

    trainset = FashionMNIST(
        "./data", train=True, download=True, transform=train_transform
    )
    testset = FashionMNIST(
        "./data", train=False, download=True, transform=test_transform
    )
    return (
        DataLoader(trainset, batch_size=32, shuffle=True),
        DataLoader(testset),
    )

def train_test_split(indices, test_size=0.2, random_state=42):
    """Split indices into train and test sets."""
    np.random.seed(random_state)
    np.random.shuffle(indices)
    split = int(np.floor(test_size * len(indices)))
    return indices[split:], indices[:split]

def train(net, trainloader, valloader, epochs, device, lr=0.001):
    """Train the model on the training set, measuring time."""
    log(INFO, "Starting training...")

    # Start measuring training time
    start_time = time.time()

    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # End measuring training time
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    avg_trainloss = running_loss / len(trainloader)

    # Evaluate on training and validation sets
    train_loss, train_acc, train_f1 = test(net, trainloader, device)
    val_loss, val_acc, val_f1 = test(net, valloader, device)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
    }

    return results, training_time

def test(net, testloader, device):
    """Evaluate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss = 0.0
    all_preds = []
    all_labels = []

    net.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate F1 score
    f1 = f1_score_torch(all_labels, all_preds, num_classes=10, average='macro')

    return loss, accuracy, f1

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate precision and recall for each class
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

