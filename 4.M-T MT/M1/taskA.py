from collections import OrderedDict
from logging import INFO
import time
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 6

class Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super(Net, self).__init__()   
        self.enc1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bottleneck = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1)       
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        skip1 = x1
        x1 = self.pool(x1)
        x2 = F.relu(self.enc2(x1))
        skip2 = x2
        x2 = self.pool(x2)
        x = F.relu(self.bottleneck(x2))
        x = F.interpolate(x, size=skip2.shape[2:], mode='nearest')
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.dec2(x))  
        x = F.interpolate(x, size=skip1.shape[2:], mode='nearest')
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.dec1(x))
        
        return self.final(x)

class NYUv2SegDataset(Dataset):
    def __init__(self, data_dir="./data", transform_rgb=None, transform_label=None):
        super().__init__()
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, "rgb_images")
        self.labels_dir = os.path.join(data_dir, "labels")        
        self.rgb_files = sorted(os.listdir(self.rgb_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
        self.transform_rgb = transform_rgb
        self.transform_label = transform_label

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        img = Image.open(rgb_path).convert("RGB")
        label_map = np.load(label_path)
        label_map_pil = Image.fromarray(label_map.astype(np.uint8), mode='L')
        
        if self.transform_rgb:
            img = self.transform_rgb(img)
        if self.transform_label:
            label_map_pil = self.transform_label(label_map_pil)
        
        label_tensor = torch.from_numpy(np.array(label_map_pil)).long()
        return img, label_tensor

# Implementazione della Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W); targets: (B, H, W)
        num_classes = logits.size(1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)
        intersection = torch.sum(probs * targets_onehot, dim=(2,3))
        union = torch.sum(probs + targets_onehot, dim=(2,3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss.mean()

# Loss combinata: CrossEntropy + Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        return ce_loss + self.dice_weight * dice_loss

def load_data():
    transform_img = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_lbl = Compose([])

    dataset = NYUv2SegDataset(
        data_dir="./data", 
        transform_rgb=transform_img,
        transform_label=transform_lbl
    )
    
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=2, shuffle=False)
    
    return train_loader, test_loader

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")
    start_time = time.time()
    net.to(device)
    
    # Usa la CombinedLoss invece della sola CrossEntropyLoss
    criterion = CombinedLoss(dice_weight=0.5, smooth=1e-6).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Le immagini sono (B, 3, H, W) e le labels (B, H, W)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")
    start_comm_time = time.time()
    
    train_loss, train_acc, train_f1 = test(net, trainloader)
    val_loss, val_acc, val_f1 = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "train_mae": 0.0,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_mae": 0.0,
    }

    gc.collect()
    return results, training_time, start_comm_time

def test(net, testloader):
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
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
            predicted_flat = predicted.view(-1)
            labels_flat = labels.view(-1)
            correct += (predicted_flat == labels_flat).sum().item()
            all_preds.append(predicted_flat)
            all_labels.append(labels_flat)

    total_pixels = sum([t.numel() for t in all_labels])
    accuracy = correct / total_pixels

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score_torch(all_labels, all_preds, num_classes=6, average='macro')
    return loss, accuracy, f1

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
    net.load_state_dict(state_dict, strict=False)