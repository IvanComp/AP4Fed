from collections import OrderedDict
from logging import INFO
import os
import time  
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LightImageNetCNN(nn.Module):
    def __init__(self) -> None:
        super(LightImageNetCNN, self).__init__()
        # Input: 256x256x3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # -> 128x128x64
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 64x64x64
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # -> 64x64x128
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 32x32x128
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # -> 32x32x256
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 16x16x256
        
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))  # -> 8x8x256
        
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 100)  # Updated to 100 classes for ImageNet100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.avg_pool(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_data(data_path, batch_size=16):
    """Load ImageNet100 dataset (training and validation sets)."""
    # Data augmentation and normalization for training
    # Different transforms for train and validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Construct paths to train and validation folders
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    
    # Create datasets
    trainset = ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
    valset = ImageFolder(
        root=val_dir,
        transform=val_transform
    )
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )
    
    valloader = DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return trainloader, valloader

def train(net, trainloader, valloader, epochs, device):
    log(INFO, "Starting training...")
    start_time = time.time()

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Step the learning rate scheduler
        scheduler.step()
        
        log(INFO, f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    # Evaluate on train and validation sets
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

    return results, training_time, time.time()

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    net.to(DEVICE)
    net.eval()  # Set the model to evaluation mode
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            all_preds.append(predicted)
            all_labels.append(labels)

    # Calculate accuracy
    accuracy = correct / len(testloader.dataset)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate F1 score
    f1 = f1_score_torch(all_labels, all_preds, num_classes=100, average='macro')
    
    return total_loss / len(testloader), accuracy, f1

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    """Calculate F1 score using PyTorch operations."""
    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_true.device)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate precision and recall for each class
    precision = torch.zeros(num_classes, device=y_true.device)
    recall = torch.zeros(num_classes, device=y_true.device)
    f1_per_class = torch.zeros(num_classes, device=y_true.device)
    
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
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)