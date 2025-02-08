from collections import OrderedDict
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from logging import INFO
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
from torchvision import models
import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i*growth_rate, growth_rate))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.convt(x)

class Net(nn.Module):
    def __init__(self, in_channels=3, growth_rate=64):
        super(Net, self).__init__()
        
        # Encoder path
        self.dense1 = DenseBlock(in_channels, growth_rate, 2)          # out: 3 + 2*64 = 131
        self.td1 = TransitionDown(131, growth_rate*2)                  # out: 128
        
        self.dense2 = DenseBlock(growth_rate*2, growth_rate*2, 2)      # out: 128 + 2*128 = 384
        self.td2 = TransitionDown(384, growth_rate*4)                  # out: 256
        
        self.dense3 = DenseBlock(growth_rate*4, growth_rate*4, 2)      # out: 256 + 2*256 = 768
        self.td3 = TransitionDown(768, growth_rate*8)                  # out: 512
        
        # Bottleneck
        self.bottleneck = DenseBlock(growth_rate*8, growth_rate*8, 2)  # out: 512 + 2*512 = 1536
        
        # Decoder path
        self.tu3 = TransitionUp(1536, growth_rate*8)                   # out: 512
        self.dense_u3 = DenseBlock(512 + 768, growth_rate*4, 2)        # out: 1280 + 2*256 = 1792
        
        self.tu2 = TransitionUp(1792, growth_rate*4)                   # out: 256
        self.dense_u2 = DenseBlock(256 + 384, growth_rate*2, 2)        # out: 640 + 2*128 = 896
        
        self.tu1 = TransitionUp(896, growth_rate*2)                    # out: 128
        self.dense_u1 = DenseBlock(128 + 131, growth_rate, 2)          # out: 259 + 2*64 = 387
        
        # Final convolution
        self.final = nn.Conv2d(387, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.dense1(x)          # 3 -> 131
        x2 = self.td1(x1)            # 131 -> 128
        x2 = self.dense2(x2)         # 128 -> 384
        x3 = self.td2(x2)            # 384 -> 256
        x3 = self.dense3(x3)         # 256 -> 768
        x4 = self.td3(x3)            # 768 -> 512
        
        # Bottleneck
        x4 = self.bottleneck(x4)     # 512 -> 1536
        
        # Decoder
        x = self.tu3(x4)             # 1536 -> 512
        x = self.dense_u3(torch.cat([x, x3], 1))  # (512 + 768) -> 1792
        x = self.tu2(x)              # 1792 -> 256
        x = self.dense_u2(torch.cat([x, x2], 1))  # (256 + 384) -> 896
        x = self.tu1(x)              # 896 -> 128
        x = self.dense_u1(torch.cat([x, x1], 1))  # (128 + 131) -> 387
        
        return self.final(x)         # 387 -> 1

class DepthEstimationDataset(Dataset):
    def __init__(self, data_dir="./data", transform_rgb=None, transform_depth=None):
        super(DepthEstimationDataset, self).__init__()
        self.rgb_dir = os.path.join(data_dir, "rgb_images")
        self.depth_dir = os.path.join(data_dir, "depth_maps")
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if not f.startswith('.')])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if not f.startswith('.')])
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
    def __len__(self):
        return len(self.rgb_files)
    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("L")
        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)
        if self.transform_depth:
            depth_img = self.transform_depth(depth_img)
        return rgb_img, depth_img

def load_data():
    transform_rgb = Compose([ToTensor(), Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    transform_depth = Compose([ToTensor()])
    dataset = DepthEstimationDataset(data_dir="./data", transform_rgb=transform_rgb, transform_depth=transform_depth)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    indices = list(range(total_len))
    random.shuffle(indices)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]
    trainset = Subset(dataset, train_indices)
    testset = Subset(dataset, test_indices)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    return trainloader, testloader

def mae_score_torch(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def f1_score_torch(y_true, y_pred, num_classes, average="macro"):
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
    if average == "macro":
        return f1_per_class.mean().item()
    elif average == "micro":
        TP = torch.diag(confusion_matrix).sum()
        FP = confusion_matrix.sum() - torch.diag(confusion_matrix).sum()
        FN = FP
        precision_micro = TP / (TP + FP + 1e-8)
        recall_micro = TP / (TP + FN + 1e-8)
        return (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)).item()
    else:
        raise ValueError("Il parametro 'average' deve essere 'macro' o 'micro'")

def test(net, testloader):
    net.to(DEVICE)
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    all_preds_quant = []
    all_labels_quant = []
    all_preds_cont = []
    all_labels_cont = []
    threshold = 0.1
    with torch.no_grad():
        for images, depths in testloader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, depths)
            total_loss += loss.item()
            abs_diff = torch.abs(outputs - depths)
            correct_pixels += (abs_diff < threshold).sum().item()
            total_pixels += depths.numel()
            outputs_cpu = outputs.detach().cpu()
            depths_cpu = depths.detach().cpu()
            all_preds_cont.append(outputs_cpu)
            all_labels_cont.append(depths_cpu)
            preds_class = (torch.clamp(outputs_cpu, 0, 1) * 9).long().squeeze(1)
            depths_class = (torch.clamp(depths_cpu, 0, 1) * 9).long().squeeze(1)
            all_preds_quant.append(preds_class)
            all_labels_quant.append(depths_class)
    avg_loss = total_loss / len(testloader)
    accuracy = correct_pixels / total_pixels
    all_preds_quant = torch.cat(all_preds_quant)
    all_labels_quant = torch.cat(all_labels_quant)
    f1 = f1_score_torch(all_labels_quant, all_preds_quant, num_classes=10, average="macro")
    all_preds_cont = torch.cat(all_preds_cont).squeeze(1)
    all_labels_cont = torch.cat(all_labels_cont).squeeze(1)
    mae = mae_score_torch(all_labels_cont, all_preds_cont)
    return avg_loss, accuracy, f1, mae

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Inizio training...")
    start_time = time.time()
    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        for images, depths in trainloader:
            images = images.to(device)
            depths = depths.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    log(INFO, f"Training completato in {training_time:.2f} secondi")
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

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.net = Net(decoder_width=1.0)
        self.trainloader, self.testloader = load_data()

    def get_parameters(self):
        return get_weights(self.net)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        epochs = int(config.get("epochs", 1))
        results, training_time, _ = train(self.net, self.trainloader, self.testloader, epochs, DEVICE)
        return get_weights(self.net), results, training_time

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy, f1, mae = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy, "f1": f1, "mae": mae}
