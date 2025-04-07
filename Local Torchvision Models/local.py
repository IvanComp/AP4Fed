import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import platform
import torchvision.models as models
import inspect
import matplotlib.pyplot as plt
import os

print(f"CPU Type: {platform.processor()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device Used: {device}")

dataset_choice = input("Select the dataset (CIFAR-10 or ImageNet100): ").lower()

if dataset_choice == "cifar-10":
    num_classes = 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

elif dataset_choice == "imagenet100":
    num_classes = 100
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        trainset = torchvision.datasets.ImageFolder(root='./imagenet100/train', transform=transform)
        testset = torchvision.datasets.ImageFolder(root='./imagenet100/val', transform=transform)
    except FileNotFoundError:
        print("Ensure ImageNet100 is downloaded and placed correctly in the 'imagenet100' folder.")
        exit()

else:
    print("Invalid dataset. Choose CIFAR-10 or ImageNet100.")
    exit()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

print("Data loading completed.")

model_names = [name for name in dir(models) if callable(getattr(models, name)) and not name.startswith('_')]
model_names_filtered = [name for name in model_names if not name.startswith('_') and not name.isupper()]

if dataset_choice == "cifar-10":
    compatible_models = [
        "alexnet", "convnext_tiny", "densenet121", "densenet161", "densenet169", "densenet201",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
        "googlenet", "inception_v3", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
        "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
        "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_y_400mf",
        "regnet_y_800mf", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "resnext50_32x4d", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_0", "squeezenet1_1",
        "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
        "wide_resnet50_2", "wide_resnet101_2","swin_t", "swin_s", "swin_b"
    ]
elif dataset_choice == "imagenet100":
    compatible_models = [
      "resnet18","resnet34","resnet50","resnet101","resnet152","densenet121","densenet161",
      "densenet169","densenet201","efficientnet_b0","efficientnet_b1","efficientnet_b2",
      "efficientnet_b3","efficientnet_b4","efficientnet_b5","efficientnet_b6","efficientnet_b7",
      "efficientnet_v2_s","efficientnet_v2_m","efficientnet_v2_l","inception_v3","mobilenet_v2",
      "mobilenet_v3_large","mobilenet_v3_small","regnet_x_16gf","regnet_x_1_6gf","regnet_x_32gf",
      "regnet_x_3_2gf","regnet_x_400mf","regnet_x_800mf","regnet_x_8gf","regnet_y_128gf",
      "regnet_y_16gf","regnet_y_1_6gf","regnet_y_32gf","regnet_y_3_2gf","regnet_y_400mf",
      "regnet_y_800mf","regnet_y_8gf","vit_b_16","vit_b_32", "vit_l_16", "vit_l_32",
      "convnext_tiny","convnext_small","convnext_base", "convnext_large","swin_t", "swin_s", "swin_b"
    ]

print("Compatible Models:")
for i, name in enumerate(compatible_models):
    if name in model_names_filtered:
        print(f"{i + 1}. {name}")

model_choice = int(input("Select the model number: ")) - 1
model_name = compatible_models[model_choice]

model_constructor = getattr(models, model_name)
constructor_args = inspect.getfullargspec(model_constructor).args

kwargs = {}
if 'num_classes' in constructor_args:
    kwargs['num_classes'] = num_classes

try:
    net = model_constructor(**kwargs).to(device)
except TypeError:
    try:
        net = model_constructor().to(device)
        try:
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes).to(device)
        except AttributeError:
            try:
                num_ftrs = net.AuxLogits.fc.in_features
                net.AuxLogits.fc = nn.Linear(num_ftrs, num_classes).to(device)
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, num_classes).to(device)
            except AttributeError:
                print(f"Model {model_name} might not be fully compatible with specified parameters or requires specific handling of the final layer. Try another model.")
                exit()
    except Exception as e:
        print(f"Error instantiating model {model_name}: {e}. Try another model.")
        exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_losses = []
val_losses = []
f1_scores = []

for epoch in range(3):
    print(f"Epoch {epoch + 1} started")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

    net.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(testloader)
    val_losses.append(val_loss)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_scores.append(f1)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}, F1-score: {f1}")
    net.train()

print('Training Finished')
# Save the model
model_save_path = f"{model_name}_{dataset_choice}_trained.pt"
torch.save(net.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Plotting F1-scores
plt.figure(figsize=(10, 5))
plt.plot(f1_scores, label='F1-score')
plt.title('F1-score')
plt.xlabel('Epochs')
plt.ylabel('F1-score')
plt.legend()
plt.show()

# Plotting training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('End of program')