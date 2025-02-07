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
num_classes = 6  # Ricorda: per NYUv2 in realtà questo valore potrebbe essere 40

class Net(nn.Module):
    def __init__(self, num_classes=6):  
        super(Net, self).__init__()
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dec_conv1 = nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = F.relu(self.enc_conv1(x))
        x1 = F.relu(self.enc_conv2(x1))
        skip = x1  
        x2 = self.pool(x1)
        x3 = F.relu(self.bottleneck_conv1(x2))
        x3 = F.relu(self.bottleneck_conv2(x3))
        # Per gestire input di dimensioni variabili, usiamo l'interpolazione per adattare le dimensioni al "skip connection"
        x4 = F.interpolate(x3, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat([x4, skip], dim=1)
        x5 = F.relu(self.dec_conv1(x_cat))
        x5 = F.relu(self.dec_conv2(x5))
        out = self.final(x5)
        return out

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

def load_data():
    transform_img = Compose([
        # Se desideri forzare una dimensione fissa, decommenta Resize/CenterCrop
        # Resize((96, 128), interpolation=Image.NEAREST),
        # CenterCrop((80, 112)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_lbl = Compose([
        # Resize((96, 128), interpolation=Image.NEAREST),
        # CenterCrop((80, 112)),
    ])

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
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels.squeeze(1))
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")
    start_comm_time = time.time()
    
    # Utilizzo esattamente dello stesso procedimento di Task1 per il test
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
            # Per la segmentazione, le uscite hanno forma (B, H, W); 
            # appiattiamo le predizioni e le etichette come in Task1
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
    # Uso della stessa funzione f1_score_torch di Task1
    f1 = f1_score_torch(all_labels, all_preds, num_classes=6, average='macro')
    return loss, accuracy, f1

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    # Creazione della matrice di confusione (identica a Task1)
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