from collections import OrderedDict
from logging import INFO
import time  # Per misurare i tempi di training
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

# Configurazione del dispositivo
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NYUv2Dataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None, target_transform=None):
        """
        Dataset personalizzato per NYUv2 per Depth Estimation.

        Args:
            rgb_dir (str): Directory contenente le immagini RGB.
            depth_dir (str): Directory contenente le mappe di profondità.
            transform (callable, optional): Trasformazioni da applicare alle immagini RGB.
            target_transform (callable, optional): Trasformazioni da applicare alle mappe di profondità.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.target_transform = target_transform
        self.rgb_images = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        self.depth_images = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        assert len(self.rgb_images) == len(self.depth_images), "Numero di immagini RGB e Depth non corrisponde."

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_images[idx])

        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')  # Assumendo che le mappe di profondità siano in scala di grigi

        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.target_transform:
            depth_image = self.target_transform(depth_image)
        else:
            depth_image = ToTensor()(depth_image)  # Default: converti in tensor

        return rgb_image, depth_image

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 5)  
        self.fc1 = nn.Linear(8 * 5 * 5, 60)  
        self.fc2 = nn.Linear(60, 42)  
        self.fc3 = nn.Linear(42, 1)  # Output modificato per regressione (1 valore di profondità)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output lineare per regressione

def load_data():
    rgb_dir = './data/rgb_images'
    depth_dir = './data/depth_maps'

    # Trasformazioni per le immagini RGB
    transform = Compose([
        Resize((32, 32)),  # Mantiene le dimensioni originali di CIFAR10
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizzazione RGB
    ])

    # Trasformazioni per le mappe di profondità
    target_transform = Compose([
        Resize((32, 32)),
        ToTensor()
    ])

    dataset = NYUv2Dataset(rgb_dir, depth_dir, transform=transform, target_transform=target_transform)
    
    # Suddivisione in training e validation (80% training, 20% validation)
    total_size = len(dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return trainloader, valloader

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")

    # Inizio misurazione del tempo di training
    start_time = time.time()

    net.to(device)  # Sposta il modello sul dispositivo (GPU se disponibile)
    criterion = torch.nn.MSELoss().to(device)  # Use MSELoss per regressione
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, depths in trainloader:
            images, depths = images.to(device), depths.to(device).view(images.size(0), -1)  # Appiattisce le mappe di profondità

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        log(INFO, f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    # Fine misurazione del tempo di training
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")
    comm_start_time = time.time()
    
    train_loss, train_mae = test(net, trainloader, device)
    val_loss, val_mae = test(net, valloader, device)

    results = {
        "train_loss": train_loss,
        "train_mae": train_mae,
        "val_loss": val_loss,
        "val_mae": val_mae,
        # Rimuovi le metriche non applicabili
    }

    return results, training_time, comm_start_time

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.MSELoss()
    net.eval()

    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for images, depths in testloader:
            images, depths = images.to(device), depths.to(device).view(images.size(0), -1)  
            outputs = net(images)
            loss = criterion(outputs, depths)
            total_loss += loss.item() * images.size(0)
            mae = torch.abs(outputs - depths).mean().item()
            total_mae += mae * images.size(0)

    average_loss = total_loss / len(testloader.dataset)
    average_mae = total_mae / len(testloader.dataset)

    return average_loss, average_mae

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
