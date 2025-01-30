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

################################################################################
#                              MODEL                                           #
################################################################################

class Net(nn.Module):
    """
    Rete semplice per semantic segmentation:
    - Due conv+pool in encoder
    - Un transpose conv (upsampling)
    - Conv finale che produce [num_classes, H, W]
    """
    def __init__(self, num_classes=895):
        super(Net, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # riduce risoluzione di 1/2

        # Decoder (upsample)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # Output: 895 possibili classi
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))      
        x = self.pool(F.relu(self.conv2(x)))  
        x = F.relu(self.up1(x))         
        x = F.relu(self.conv3(x))       
        x = self.out_conv(x)
        return x

class NYUv2SegDataset(Dataset):

    def __init__(self, data_dir="./data", transform_rgb=None, transform_label=None):
        super().__init__()
        self.data_dir = data_dir
        
        self.rgb_dir = os.path.join(data_dir, "rgb_images")
        self.labels_dir = os.path.join(data_dir, "labels")
        
        self.rgb_files = sorted(os.listdir(self.rgb_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
        
        assert len(self.rgb_files) == len(self.label_files), \
            "Mismatch tra numero di immagini RGB e file di label"
        
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
            label_tensor = label_map_pil[0, :, :].long()  
        else:
            label_tensor = torch.from_numpy(label_map).long()

        return img, label_tensor

def load_data():
    transform_img = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_lbl = Compose([
        Resize((64, 64), interpolation=Image.NEAREST),
        ToTensor(),
    ])

    dataset = NYUv2SegDataset(
        data_dir="./data", 
        transform_rgb=transform_img,
        transform_label=transform_lbl
    )

    # Split train/test
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False)
    return train_loader, test_loader

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")
    start_time = time.time()

    net.to(device)
    # CrossEntropyLoss su output [B, classes, H, W] e label [B, H, W]
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Predizione
            outputs = net(images)  # [B, 895, H, W]
            # Loss pixel-wise
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    comm_start_time = time.time()

    # Calcoliamo metriche su train e val
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
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    # Per calcolare accuracy e F1, accumuliamo i pixel
    all_preds = []
    all_labels = []
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)  # [B, H, W]
            outputs = net(images)       # [B, 895, H, W]

            # Loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Predizione pixel-wise
            # outputs ha shape [B, 895, H, W]
            # facciamo argmax su dim=1
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Calcoliamo accuratezza pixel
            correct_pixels += (preds == labels).sum().item()
            total_pixels   += labels.numel()

            # Conserviamo per F1 globale
            # Flatten per formare dei vettori 1D pixel-wise
            all_preds.append(preds.view(-1))
            all_labels.append(labels.view(-1))

    avg_loss = total_loss / len(testloader)
    pixel_accuracy = correct_pixels / total_pixels

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    # Adoperiamo un confusion matrix 895×895
    # e poi calcoliamo la macro-F1.
    f1_val = f1_score_torch(all_labels, all_preds, num_classes=895, average='macro')

    return avg_loss, pixel_accuracy, f1_val

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    """
    Calcoliamo la confusion matrix e la F1 "macro" su tutti i pixel,
    assumendo che y_true e y_pred siano vettori 1D di pixel totali.
    """
    device = y_true.device
    cm = torch.zeros(num_classes, num_classes, device=device)

    for t, p in zip(y_true, y_pred):
        cm[t.long(), p.long()] += 1

    # Calcolo di precision, recall, F1 per classe
    precision = torch.zeros(num_classes, device=device)
    recall = torch.zeros(num_classes, device=device)
    f1_per_class = torch.zeros(num_classes, device=device)

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP

        precision[i] = TP / (TP + FP + 1e-8)
        recall[i] = TP / (TP + FN + 1e-8)
        f1_per_class[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)

    if average == 'macro':
        f1 = f1_per_class.mean().item()
    elif average == 'micro':
        TP = torch.diag(cm).sum()
        total = cm.sum()
        FP = total - TP
        FN = FP
        precision_micro = TP / (TP + FP + 1e-8)
        recall_micro = TP / (TP + FN + 1e-8)
        f1 = (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)).item()
    else:
        raise ValueError("Average must be 'macro' or 'micro'")

    return f1

def get_weights(net):
    """Estrai i pesi come array NumPy (per Flower)."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Imposta i pesi a partire da parametri (NumPy) (per Flower)."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
