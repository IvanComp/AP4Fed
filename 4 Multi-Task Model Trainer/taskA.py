from collections import OrderedDict
from logging import INFO
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from flwr.common.logger import log

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset per la Segmentazione: legge da data/rgb_images/*.png e data/labels/*.npy
class SegmentationDataset(Dataset):
    def __init__(self, rgb_dir, label_dir, transform=None):
        super().__init__()
        self.rgb_dir = rgb_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = []
        # In base al tuo script, le immagini si chiamano rgb_0.png, rgb_1.png ... e i file .npy label_0.npy ...
        # Supponendo di aver generato 101 file (da 0 a 100), potresti enumerarli o controllare esistenza
        # Qui, per semplicità, assumo che esistano e siano numerati consecutivamente
        for idx in range(101):
            # Se preferisci fermarti prima, basterà controllare se i file esistono
            self.image_paths.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        index = self.image_paths[idx]
        rgb_path = f"{self.rgb_dir}/rgb_{index}.png"
        lbl_path = f"{self.label_dir}/label_{index}.npy"

        # Carica immagine RGB
        image = Image.open(rgb_path).convert("RGB")

        # Carica label come array
        label_array = np.load(lbl_path)
        # label_array potrebbe essere shape (H, W). Convertiamola in torch Tensor
        label_tensor = torch.from_numpy(label_array).long()

        if self.transform:
            # Applichiamo la trasformazione solo all'immagine
            image = self.transform(image)

        # L'immagine è un Tensor di shape [3, H, W], la label è [H, W] con i "class index" per ogni pixel
        return image, label_tensor

# Rete molto semplificata per la Segmentazione. Output = n. classi, dimensione spaziale = H, W
# Qui per esempio si assume che ci siano 13 classi (o 40). Cambia in base al dataset. Metto 13 come esempio.
class Net(nn.Module):
    def __init__(self, num_classes=13):
        super(Net, self).__init__()
        # Modello di esempio simile a un piccolo UNet semplificato
        # Per restare "compatibile" con la logica di un forward, useremo conv down e conv up
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.enc_conv1(x))
        x2 = self.pool(F.relu(self.enc_conv2(x1)))
        x3 = F.interpolate(x2, scale_factor=2, mode="nearest")  # upsamping
        x4 = F.relu(self.dec_conv1(x3))
        x5 = self.dec_conv2(x4)
        return x5  # shape [batch, num_classes, H, W]

def load_data():
    # Caricamento dataset per la Segmentazione
    # Esempio di trasformazione minima: ridimensionamento e conversione a Tensor
    transform = transforms.Compose([
        transforms.Resize((240, 320)),   # Esempio di dimensione fissa, adattala se preferisci
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(rgb_dir="data/rgb_images",
                                  label_dir="data/labels",
                                  transform=transform)

    # Split train/test (80% train, 20% test)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_test = n - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
    return train_loader, test_loader

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")
    start_time = time.time()

    net.to(device)
    # Per la segmentazione multi-classe useremo la CrossEntropy2D
    # net ha come output dimensione [batch, num_classes, H, W], label shape [batch, H, W]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    # Calcoliamo metrica su train e val
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
    return results, training_time

def test(net, testloader):
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    net.eval()

    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Calcoliamo la pixel-accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(testloader)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    # Calcoliamo un F1 "macro" su tutte le classi
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    f1 = f1_score_torch(all_labels, all_preds)

    return avg_loss, pixel_accuracy, f1

def f1_score_torch(y_true, y_pred, num_classes=13):
    # Creiamo confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
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
        f1_per_class[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

    return f1_per_class.mean().item()

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)