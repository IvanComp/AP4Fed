from collections import OrderedDict
from logging import INFO
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 5)
        self.fc1 = nn.Linear(8 * 5 * 5, 60)
        self.fc2 = nn.Linear(60, 42)
        self.fc3 = nn.Linear(42, 1)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def load_data():
    transform_rgb = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_depth = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    rgb_dir = './data/rgb_images'
    depth_dir = './data/depth_maps'
    rgb_images = sorted(os.listdir(rgb_dir))
    depth_maps = sorted(os.listdir(depth_dir))

    assert len(rgb_images) == len(depth_maps), "Mismatch between RGB and depth files."

    indices = list(range(len(rgb_images)))
    random.shuffle(indices)
    train_size = int(0.8 * len(indices))  
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = [(os.path.join(rgb_dir, rgb_images[i]), os.path.join(depth_dir, depth_maps[i])) for i in train_indices]
    test_data = [(os.path.join(rgb_dir, rgb_images[i]), os.path.join(depth_dir, depth_maps[i])) for i in test_indices]

    def dataset_generator(data, transform_rgb, transform_depth):
        for idx, (rgb_path, depth_path) in enumerate(data):
            rgb_image = Image.open(rgb_path).convert("RGB")
            depth_map = Image.open(depth_path).convert("L")  # Grayscale

            if transform_rgb and transform_depth:
                rgb_image = transform_rgb(rgb_image)
                depth_map = transform_depth(depth_map)

            # Debugging output per verificare le dimensioni e i dati
            if idx == 0:  # Solo per il primo elemento per non sovraccaricare l'output
                print(f"Sample RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                print(f"Sample depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")
                print(f"RGB image (first pixel): {rgb_image[:, 0, 0]}")
                print(f"Depth map (first pixel): {depth_map[0, 0]}")

            yield depth_map.unsqueeze(0), depth_map  # Aggiungi un canale

    trainloader = DataLoader(list(dataset_generator(train_data, transform_rgb, transform_depth)), batch_size=32, shuffle=True)
    testloader = DataLoader(list(dataset_generator(test_data, transform_rgb, transform_depth)), batch_size=32, shuffle=False)

    return trainloader, testloader

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")

    start_time = time.time()

    net.to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            print(f"Epoch {epoch}, Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}, dtype: {images.dtype}")
            print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    train_loss, train_acc, _ = test(net, trainloader)
    val_loss, val_acc, _ = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }

    return results, training_time

def test(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.MSELoss()
    loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()

    return loss, None, None

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    pass  # Non necessario per DepthNet

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
