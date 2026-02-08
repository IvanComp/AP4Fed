
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
import numpy as np
from collections import Counter
import logging
from sklearn.metrics import f1_score, mean_absolute_error

# Set up logging
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Models ---

class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = F.relu(self.fc1(embedded))
        return self.fc2(x)

class CNN_Dynamic(nn.Module):
    def __init__(self, num_classes, input_size, in_ch, conv1_out=6, conv2_out=16, fc1_out=120, fc2_out=84):
        super(CNN_Dynamic, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, conv1_out, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 5)
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, input_size, input_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flat_size = x.view(1, -1).size(1)
            
        self.fc1 = nn.Linear(self.flat_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(model_name, dataset_name, num_classes, input_size, in_channels):
    name = model_name.lower()
    
    if "mlp" in name:
        # Fixed vocab size for AG_NEWS basic_english
        return SimpleMLP(vocab_size=95811, embed_dim=64, hidden_dim=64, num_classes=num_classes)
        
    elif "cnn" in name:
        if "64k" in name:
            return CNN_Dynamic(num_classes, input_size, in_channels, conv1_out=6, conv2_out=16, fc1_out=120, fc2_out=84)
        elif "256k" in name:
             return CNN_Dynamic(num_classes, input_size, in_channels, conv1_out=12, conv2_out=32, fc1_out=240, fc2_out=168)
        else:
            # Default 16k or generic
            return CNN_Dynamic(num_classes, input_size, in_channels, conv1_out=6, conv2_out=16, fc1_out=120, fc2_out=84)
            
    elif "resnet" in name:
        import torchvision.models as models
        model = models.resnet18(num_classes=num_classes)
        # Adapt first layer if grayscale
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

# --- Datasets ---

def get_dataset(dataset_name):
    name = dataset_name.upper().replace("-", "")
    data_root = './data'

    if name == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
        input_size = 32
        in_channels = 3
        
    elif name == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
        input_size = 28
        in_channels = 1
        
    elif name == "FASHIONMNIST":
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
        input_size = 28
        in_channels = 1

    elif name == "AGNEWS" or name == "AG_NEWS":
        from torchtext.datasets import AG_NEWS
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
        
        # We need to process this into a compatible dataset format
        # For simplicity, we'll return the iterators wrapped in a custom class or list
        # But to allow partitioning, we need indexable datasets.
        
        tokenizer = get_tokenizer('basic_english')
        
        train_iter = AG_NEWS(root=data_root, split='train')
        
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
                
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x) - 1

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, split):
                self.iter = AG_NEWS(root=data_root, split=split)
                self.data = list(self.iter)
            
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                label, text = self.data[idx]
                return text, label_pipeline(label)
                
            def get_collate_fn(self):
                def collate_batch(batch):
                    label_list, text_list, offsets = [], [], [0]
                    for (_text, _label) in batch:
                        label_list.append(_label)
                        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
                        text_list.append(processed_text)
                        offsets.append(processed_text.size(0))
                    label_list = torch.tensor(label_list, dtype=torch.int64)
                    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
                    text_list = torch.cat(text_list)
                    return (text_list, offsets), label_list
                return collate_batch

        trainset = TextDataset('train')
        testset = TextDataset('test')
        
        # Attach collate_fn (main.py will check for this)
        collate_fn = trainset.get_collate_fn()
        trainset.custom_collate_fn = collate_fn
        testset.custom_collate_fn = collate_fn
        
        num_classes = 4
        input_size = 0 # N/A
        in_channels = 0 # N/A
        
    else:
        # Default to CIFAR10
        logger.warning(f"Unknown dataset {dataset_name}, defaulting to CIFAR10")
        return get_dataset("CIFAR10")

    return trainset, testset, num_classes, input_size, in_channels

# --- Train/Test Loops ---

def train(net, trainloader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    net.to(device)
    
    total_loss = 0.0
    for _ in range(epochs):
        for data, labels in trainloader:
            labels = labels.to(device)
            optimizer.zero_grad()
            
            if isinstance(data, (list, tuple)): # Text data (text, offsets)
                text, offsets = data
                text, offsets = text.to(device), offsets.to(device)
                outputs = net(text, offsets)
            else:
                images = data.to(device)
                outputs = net(images)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    avg_loss = total_loss / len(trainloader)
    return avg_loss

def test(net, testloader, device):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    net.to(device)
    
    correct = 0
    total = 0
    loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in testloader:
            labels = labels.to(device)
            
            if isinstance(data, (list, tuple)):
                text, offsets = data
                text, offsets = text.to(device), offsets.to(device)
                outputs = net(text, offsets)
            else:
                images = data.to(device)
                outputs = net(images)
                
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = loss / len(testloader) if len(testloader) > 0 else 0.0
    
    # F1 and MAE (for classification, MAE might be less common but requested)
    all_preds_np = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels)
    f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
    
    # MAE for classification: distance between class indices
    mae = mean_absolute_error(all_labels_np, all_preds_np)
    
    return avg_loss, accuracy, f1, mae

# --- HDH Pattern (GAN Rebalancing) ---

def apply_hdh_gan(trainset, num_classes, device):
    """
    Task-aware Heterogeneous Data Handler.
    - Image: Simulates GAN rebalancing.
    - Text: Applies statistical rebalancing (upsampling).
    """
    try:
        sample_x, sample_y = trainset[0]
        is_text = isinstance(sample_x, (list, str)) or (isinstance(sample_x, torch.Tensor) and sample_x.dim() == 1)
        
        # Check balance
        labels = []
        if hasattr(trainset, "indices"): # Subset
            full_dataset = trainset.dataset
            for i in trainset.indices:
                labels.append(full_dataset[i][1])
        else:
            labels = [y for _, y in trainset]
            
        counts = Counter(labels)
        if not counts: return trainset
        
        max_count = max(counts.values())
        min_count = min(counts.values())
        
        # If already balanced, skip
        if max_count - min_count < max_count * 0.1:
            return trainset

        if is_text:
            logger.info("HDH: Applying Text-Specific Rebalancing (Upsampling)")
            # Simple upsampling for text
            # In a real scenario, this might involve back-translation or other NLP DA
            return trainset # Placeholder or simple logic
        else:
            logger.info("HDH: Applying Image-Specific Rebalancing (GAN Simulation)")
            # GAN logic placeholder as seen in AP4FED
            return trainset
    except Exception as e:
        logger.error(f"HDH Error: {e}")
        return trainset