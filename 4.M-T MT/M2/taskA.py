from collections import OrderedDict
from logging import INFO
from collections import Counter
import time  
import os
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision.datasets import CIFAR10  # Non utilizzato, ma lasciato per non rompere l'interfaccia
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image
from torchvision.models import densenet121

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Abbiamo scelto una architettura molto semplice per accelerare il training:
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # Encoder: 3 convoluzioni con pooling progressivo
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # da 256x256 a 256x256, 16 canali
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 256x256 -> 128x128
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 128x128, 32 canali
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 64x64, 64 canali
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                # 64x64 -> 32x32
        )
        # Decoder: 3 upsampling per riportare la risoluzione a 256x256
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32 -> 64x64
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64 -> 128x128
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128x128 -> 256x256
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        out = self.decoder(features)
        # Assicuriamo l'output abbia la stessa dimensione spaziale dell'input
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# Dataset per depth estimation (non modificato)
class DepthEstimationDataset(Dataset):
    def __init__(self, data_dir="./data", transform_rgb=None, transform_depth=None):
        super().__init__()
        self.rgb_dir = f"{data_dir}/rgb_images"
        self.depth_dir = f"{data_dir}/depth_maps"
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if not f.startswith('.')])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if not f.startswith('.')])
        assert len(self.rgb_files) == len(self.depth_files), "Number of RGB and depth images must be the same"
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = f"{self.rgb_dir}/{self.rgb_files[idx]}"
        depth_path = f"{self.depth_dir}/{self.depth_files[idx]}"
        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("L")  
        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)
        if self.transform_depth:
            depth_img = self.transform_depth(depth_img)
        return rgb_img, depth_img

# La funzione load_data rimane invariata
def load_data():
    # Riduciamo la risoluzione a 256x256 per contenere l'uso della memoria
    transform_rgb = Compose([
        Resize((256,256), interpolation=Image.BILINEAR),
        ToTensor(),
        Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    transform_depth = Compose([
        Resize((256,256), interpolation=Image.NEAREST),
        ToTensor()  # Restituisce tensore [1,H,W] con valori in [0,1]
    ])
    dataset = DepthEstimationDataset(data_dir="./data", transform_rgb=transform_rgb, transform_depth=transform_depth)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    indices = list(range(total_len))
    random.shuffle(indices)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]
    trainset = Subset(dataset, train_indices)
    testset  = Subset(dataset, test_indices)
    # Batch size ridotto (ad es. 4)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader  = DataLoader(testset, batch_size=4, shuffle=False)
    return trainloader, testloader

# Funzione per il calcolo del MAE (Mean Absolute Error)
def mae_score_torch(y_true, y_pred):
    eps = 1e-6
    # Calcola il Mean Absolute Percentage Error (MAPE)
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps))).item() * 100

# Il metodo test è aggiornato per calcolare loss, accuracy, F1 e MAE.
def test(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    all_preds_quant = []
    all_labels_quant = []
    all_preds_cont = []
    all_labels_cont = []
    # Soglia per l'accuracy (pixel con errore assoluto < 0.1 considerati corretti)
    threshold = 0.1
    with torch.no_grad():
        for images, depths in testloader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, depths)
            total_loss += loss.item()
            # Accuracy pixel-wise
            abs_diff = torch.abs(outputs - depths)
            correct_pixels += (abs_diff < threshold).sum().item()
            total_pixels += depths.numel()
            # Conserviamo le predizioni continue per il calcolo del MAE
            all_preds_cont.append(outputs.cpu())
            all_labels_cont.append(depths.cpu())
            # Per il F1 quantizziamo in 10 classi (classe = int(pixel*9))
            preds_class = (torch.clamp(outputs, 0, 1) * 9).long().squeeze(1)
            depths_class = (torch.clamp(depths, 0, 1) * 9).long().squeeze(1)
            all_preds_quant.append(preds_class.cpu())
            all_labels_quant.append(depths_class.cpu())
    avg_loss = total_loss / len(testloader)
    accuracy = correct_pixels / total_pixels
    # Calcolo F1 score (come già presente)
    all_preds_quant = torch.cat(all_preds_quant)
    all_labels_quant = torch.cat(all_labels_quant)
    f1 = f1_score_torch(all_labels_quant, all_preds_quant, num_classes=10, average='macro')
    # Calcolo MAE sui valori continui
    all_preds_cont = torch.cat(all_preds_cont).squeeze(1)
    all_labels_cont = torch.cat(all_labels_cont).squeeze(1)
    mae = mae_score_torch(all_labels_cont, all_preds_cont)
    return avg_loss, accuracy, f1, mae

# Il metodo train viene aggiornato per includere anche il MAE nei risultati.
def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")
    start_time = time.time()
    net.to(device) 
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, depths in trainloader:
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")
    comm_start_time = time.time()
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
    return results, training_time, comm_start_time

# La funzione f1_score_torch rimane invariata
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
