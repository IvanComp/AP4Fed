from collections import OrderedDict
from logging import INFO
import time  
import os
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
from PIL import Image
from torchvision import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        x_cat = torch.cat([up_x, concat_with], dim=1)
        x_conv = self.convA(x_cat)
        x_act = self.leakyreluA(x_conv)
        x_conv2 = self.convB(x_act)
        return self.leakyreluB(x_conv2)

class Decoder(nn.Module):
    def __init__(self, num_features=1024, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(skip_input=features + 1024, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 512, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 256, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)
        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0 = features[4]
        x_block1 = features[5]
        x_block2 = features[7]
        x_block3 = features[9]
        x_block4 = features[12]
        x_d0 = self.conv2(F.relu(x_block4))
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet121()

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        out = self.decoder(features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

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

def load_data():
    transform_rgb = Compose([
        Resize((240, 320), interpolation=Image.BILINEAR),  
        CenterCrop((228, 304)),                              
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_depth = Compose([
        Resize((240, 320), interpolation=Image.NEAREST),
        CenterCrop((228, 304)),
        ToTensor()
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
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True)
    testloader  = DataLoader(testset, batch_size=2, shuffle=False)
    return trainloader, testloader

def mae_score_torch(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

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
            all_preds_cont.append(outputs.cpu())
            all_labels_cont.append(depths.cpu())
            preds_class = (torch.clamp(outputs, 0, 1) * 9).long().squeeze(1)
            depths_class = (torch.clamp(depths, 0, 1) * 9).long().squeeze(1)
            all_preds_quant.append(preds_class.cpu())
            all_labels_quant.append(depths_class.cpu())
    avg_loss = total_loss / len(testloader)
    accuracy = correct_pixels / total_pixels
    all_preds_quant = torch.cat(all_preds_quant)
    all_labels_quant = torch.cat(all_labels_quant)
    f1 = f1_score_torch(all_labels_quant, all_preds_quant, num_classes=10, average='macro')
    all_preds_cont = torch.cat(all_preds_cont).squeeze(1)
    all_labels_cont = torch.cat(all_labels_cont).squeeze(1)
    mae = mae_score_torch(all_labels_cont, all_preds_cont)
    return avg_loss, accuracy, f1, mae

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
