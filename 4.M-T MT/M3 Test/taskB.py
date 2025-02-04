from collections import OrderedDict
from logging import INFO
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, num_classes=40):
        super(Net, self).__init__()
        
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.pool = nn.MaxPool2d(2, 2)  
        
        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.bottleneck = conv_block(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))   
        b = self.bottleneck(self.pool(e3))       
        d3 = self.dec3(torch.cat((self.up3(b), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        
        return self.final(d1)

class NYUv2SegDataset(Dataset):
    def __init__(self, data_dir="./data", transform_rgb=None, transform_label=None):
        super().__init__()
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, "rgb_images")
        self.labels_dir = os.path.join(data_dir, "labels")
        
        self.rgb_files = sorted(os.listdir(self.rgb_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
        
        #assert len(self.rgb_files) == len(self.label_files), "Mismatch tra numero di immagini RGB e file label"
        
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
        Resize((240, 320), interpolation=Image.NEAREST),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_lbl = Resize((240, 320), interpolation=Image.NEAREST)

    dataset = NYUv2SegDataset(
        data_dir="./data", 
        transform_rgb=transform_img,
        transform_label=transform_lbl
    )
    
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")

    start_time = time.time()
    net.to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  
    
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
    comm_start_time = time.time()

    train_loss, train_acc, train_f1 = test(net, trainloader)
    val_loss, val_acc, val_f1 = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
    }

    return results, training_time, comm_start_time

def test(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)  
            correct_pixels += (preds == labels).sum().item()
            total_pixels   += labels.numel()

            all_preds.append(preds.view(-1))
            all_labels.append(labels.view(-1))

    avg_loss = total_loss / len(testloader)
    pixel_accuracy = correct_pixels / total_pixels

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1_val = f1_score_torch(all_labels, all_preds, num_classes=40, average='macro')

    return avg_loss, pixel_accuracy, f1_val

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    confusion_matrix = torch.zeros(num_classes, num_classes, device=DEVICE)

    for t, p in zip(y_true, y_pred):
        confusion_matrix[t.long(), p.long()] += 1

    precision = torch.zeros(num_classes, device=DEVICE)
    recall = torch.zeros(num_classes, device=DEVICE)
    f1_per_class = torch.zeros(num_classes, device=DEVICE)

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP

        precision[i] = TP / (TP + FP + 1e-8)
        recall[i] = TP / (TP + FN + 1e-8)
        f1_per_class[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)

    return f1_per_class.mean().item()

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)