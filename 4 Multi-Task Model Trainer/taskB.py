import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import os
from logging import INFO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from flwr.common.logger import log

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset per la Depth Estimation: input = immagine RGB, label = immagine di profondità (grayscale)
class DepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform_rgb=None, transform_depth=None):
        super().__init__()
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.image_paths = []
        for idx in range(101):
            self.image_paths.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        index = self.image_paths[idx]
        rgb_path = f"{self.rgb_dir}/rgb_{index}.png"
        depth_path = f"{self.depth_dir}/depth_{index}.png"

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")  # grayscale

        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        # depth è un singolo canale. Usiamo float per la regression. Normalizziamo in [0,1] se vogliamo
        depth_tensor = depth.float()  # shape [1, H, W]
        return rgb, depth_tensor

# Rete semplificata per la Depth Estimation: input=3 canali, output=1 canale con la stima di profondità
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = self.pool(F.relu(self.conv3(x2)))
        x4 = self.up(x3)
        x5 = F.relu(self.conv4(x4))
        x6 = self.up(x5)
        x7 = F.relu(self.conv5(x6))
        out = self.out(x7)
        return out  # [batch,1,H,W], stima della mappa di profondità

def load_data():
    # Carica dataset di Depth. Normalizziamo l'RGB in [0,1], e la depth in [0,1].
    from torchvision import transforms
    transform_rgb = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor()
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor()
    ])

    dataset = DepthDataset(rgb_dir="data/rgb_images",
                           depth_dir="data/depth_maps",
                           transform_rgb=transform_rgb,
                           transform_depth=transform_depth)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_test = n - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
    return train_loader, test_loader

def train(net, trainloader, valloader, epochs, device, lr=0.001):
    log(INFO, "Starting training...")
    start_time = time.time()

    net.to(device)
    # Usando MSE come loss di base
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for rgb, depth in trainloader:
            rgb, depth = rgb.to(device), depth.to(device)
            optimizer.zero_grad()
            outputs = net(rgb)
            # outputs [batch,1,H,W], depth [batch,1,H,W]
            loss = criterion(outputs, depth)
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    # Calcolo delle metriche su train e val
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
    net.to(device)
    criterion = nn.MSELoss()
    net.eval()

    total_loss = 0.0
    count = 0
    # Per “accuracy” e “f1”, non essendo una classificazione, restituiamo valori simbolici (o potremmo inventare metriche)
    with torch.no_grad():
        for rgb, depth in testloader:
            rgb, depth = rgb.to(device), depth.to(device)
            outputs = net(rgb)
            loss = criterion(outputs, depth)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else 0.0
    dummy_accuracy = 0.0
    dummy_f1 = 0.0
    return avg_loss, dummy_accuracy, dummy_f1

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)