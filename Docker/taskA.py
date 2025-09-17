import os
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import OrderedDict, Counter, defaultdict
from logging import INFO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset, Dataset
from flwr.common.logger import log
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, KMNIST, OxfordIIITPet
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose, ToPILImage
import torchvision.models as models
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.trainer import Trainer
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss
from torch.utils.data import TensorDataset, ConcatDataset
from collections import Counter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from collections import Counter
import random
from sklearn.metrics import f1_score
    
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False
global DATASET_TYPE, DATASET_NAME
DATASET_TYPE = ""
DATASET_NAME = ""

AVAILABLE_DATASETS = {
    "CIFAR10": {
        "class": CIFAR10,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "channels": 3,
        "num_classes": 10
    },
    "CIFAR100": {
        "class": CIFAR100,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "channels": 3,
        "num_classes": 100
    },
    "MNIST": {
        "class": MNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "FashionMNIST": {
        "class": FashionMNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "KMNIST": {
        "class": KMNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "ImageNet100": {
        "class": None,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "channels": 3,
        "num_classes": 10
    },
    "OXFORDIIITPET": {
        "class": OxfordIIITPet,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "channels": 3,
        "num_classes": 37
    }
}

class TensorLabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        return x, y

def get_valid_downscale_size(size: int) -> int:
    power = 32
    while power * 2 <= size and power * 2 <= 128:
        power *= 2
    return power

def normalize_dataset_name(name: str) -> str:
    name_clean = name.replace("-", "").upper()
    if name_clean == "CIFAR10":
        return "CIFAR10"
    elif name_clean == "CIFAR100":
        return "CIFAR100"
    elif name_clean == "IMAGENET100":
        return "ImageNet100"
    elif name_clean == "MNIST":
        return "MNIST"
    elif name_clean == "FASHIONMNIST":
        return "FashionMNIST"
    elif name_clean == "FMNIST":
        return "FMNIST"
    elif name_clean == "KMNIST":
        return "KMNIST"
    elif name_clean == "OXFORDIIITPET":
        return "OXFORDIIITPET"
    else:
        return name
    
current_dir = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
    for pattern_name, pattern_info in configJSON["patterns"].items():
        if pattern_info["enabled"]:
            if pattern_name == "client_selector":
                CLIENT_SELECTOR = True
            elif pattern_name == "client_cluster":
                CLIENT_CLUSTER = True
            elif pattern_name == "message_compressor":
                MESSAGE_COMPRESSOR = True
            elif pattern_name == "multi-task_model_trainer":
                MULTI_TASK_MODEL_TRAINER = True
            elif pattern_name == "heterogeneous_data_handler":
                HETEROGENEOUS_DATA_HANDLER = True
    ds = configJSON.get("dataset") or configJSON["client_details"][0].get("dataset", None)
    if ds is None:
        raise ValueError("Il file di configurazione non specifica il dataset né tramite la chiave 'dataset' né in 'client_details'.")
    DATASET_NAME = normalize_dataset_name(ds)
    DATASET_TYPE = configJSON["client_details"][0].get("data_distribution_type", "")
    EPOCHS = configJSON["client_details"][0]["epochs"]


class CNN_Dynamic(nn.Module):
    def __init__(
        self, num_classes: int, input_size: int, in_ch: int,
        conv1_out: int, conv2_out: int, fc1_out: int, fc2_out: int
    ) -> None:
        super(CNN_Dynamic, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, conv1_out, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        dummy = torch.zeros(1, in_ch, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flat_size = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
def get_weight_class_dynamic(model_name: str):
    weight_mapping = {
        "cnn": None,
        "alexnet": "AlexNet_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
        "convnext_base": "ConvNeXt_Base_Weights",
        "convnext_large": "ConvNeXt_Large_Weights",
        "densenet121": "DenseNet121_Weights",
        "densenet161": "DenseNet161_Weights",
        "densenet169": "DenseNet169_Weights",
        "densenet201": "DenseNet201_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "efficientnet_b1": "EfficientNet_B1_Weights",
        "efficientnet_b2": "EfficientNet_B2_Weights",
        "efficientnet_b3": "EfficientNet_B3_Weights",
        "efficientnet_b4": "EfficientNet_B4_Weights",
        "efficientnet_b5": "EfficientNet_B5_Weights",
        "efficientnet_b6": "EfficientNet_B6_Weights",
        "efficientnet_b7": "EfficientNet_B7_Weights",
        "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
        "efficientnet_v2_m": "EfficientNet_V2_M_Weights",
        "efficientnet_v2_l": "EfficientNet_V2_L_Weights",
        "googlenet": "GoogLeNet_Weights",
        "inception_v3": "Inception_V3_Weights",
        "mnasnet0_5": "MnasNet0_5_Weights",
        "mnasnet0_75": "MnasNet0_75_Weights",
        "mnasnet1_0": "MnasNet1_0_Weights",
        "mnasnet1_3": "MnasNet1_3_Weights",
        "mobilenet_v2": "MobileNet_V2_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
        "regnet_x_400mf": "RegNet_X_400MF_Weights",
        "regnet_x_800mf": "RegNet_X_800MF_Weights",
        "regnet_x_1_6gf": "RegNet_X_1_6GF_Weights",
        "regnet_x_16gf": "RegNet_X_16GF_Weights",
        "regnet_x_32gf": "RegNet_X_32GF_Weights",
        "regnet_x_3_2gf": "RegNet_X_3_2GF_Weights",
        "regnet_x_8gf": "RegNet_X_8GF_Weights",
        "regnet_y_400mf": "RegNet_Y_400MF_Weights",
        "regnet_y_800mf": "RegNet_Y_800MF_Weights",
        "regnet_y_128gf": "RegNet_Y_128GF_Weights",
        "regnet_y_16gf": "RegNet_Y_16GF_Weights",
        "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
        "regnet_y_32gf": "RegNet_Y_32GF_Weights",
        "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights",
        "regnet_y_8gf": "RegNet_Y_8GF_Weights",
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
        "resnext50_32x4d": "ResNeXt50_32X4D_Weights",
        "shufflenet_v2_x0_5": "ShuffleNet_V2_x0_5_Weights",
        "shufflenet_v2_x1_0": "ShuffleNet_V2_x1_0_Weights",
        "squeezenet1_0": "SqueezeNet1_0_Weights",
        "squeezenet1_1": "SqueezeNet1_1_Weights",
        "vgg11": "VGG11_Weights",
        "vgg11_bn": "VGG11_BN_Weights",
        "vgg13": "VGG13_Weights",
        "vgg13_bn": "VGG13_BN_Weights",
        "vgg16": "VGG16_Weights",
        "vgg16_bn": "VGG16_BN_Weights",
        "vgg19": "VGG19_Weights",
        "vgg19_bn": "VGG19_BN_Weights",
        "wide_resnet50_2": "Wide_ResNet50_2_Weights",
        "wide_resnet101_2": "Wide_ResNet101_2_Weights",
        "swin_t": "Swin_T_Weights",
        "swin_s": "Swin_S_Weights",
        "swin_b": "Swin_B_Weights",
        "vit_b_16": "ViT_B_16_Weights",
        "vit_b_32": "ViT_B_32_Weights",
        "vit_l_16": "ViT_L_16_Weights",
        "vit_l_32": "ViT_L_32_Weights"
    }
    model_name = model_name.lower()
    weight_class_name = weight_mapping.get(model_name, None)
    if weight_class_name is not None:
        return getattr(models, weight_class_name, None)
    return None

def get_dynamic_model(num_classes: int, model_name: str = None, pretrained: bool = True) -> nn.Module:
    if model_name is None:
        with open(config_file, 'r') as f:
            configJSON = json.load(f)
        model_name = configJSON["client_details"][0].get("model")
    name = model_name.strip().lower().replace("-", "_").replace(" ", "_")

    # cnn 16k
    if name in ("cnn_16k", "cnn16k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "MNIST": 28, "KMNIST": 28, "FMNIST": 28,
            "ImageNet100": 224, "OXFORDIIITPET": 224
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=3, conv2_out=8,
            fc1_out=60, fc2_out=42
        )
    # cnn 64k
    if name in ("cnn_64k", "cnn64k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "MNIST": 28, "KMNIST": 28, "FMNIST": 28,
            "ImageNet100": 224, "OXFORDIIITPET": 224
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=6, conv2_out=16,
            fc1_out=120, fc2_out=84
        )
    # cnn 256k
    if name in ("cnn_256k", "cnn256k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "MNIST": 28, "KMNIST": 28, "FMNIST": 28,
            "ImageNet100": 224, "OXFORDIIITPET": 224
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=12, conv2_out=32,
            fc1_out=240, fc2_out=168
        )

    if not hasattr(models, name):
        raise ValueError(f"Modello '{model_name}' non in torchvision.models")
    constructor = getattr(models, name)

    weight_cls = get_weight_class_dynamic(name)
    if pretrained and weight_cls and hasattr(weight_cls, "DEFAULT"):
        model = constructor(weights=weight_cls.DEFAULT, progress=False)
    else:
        model = constructor(weights=None, progress=False)

    if hasattr(model, 'fc'):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(in_f, num_classes)
        )

    elif hasattr(model, 'head'):
        in_f = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(in_f, num_classes)
        )

    elif hasattr(model, 'classifier'):
        cls = model.classifier
        if isinstance(cls, nn.Sequential):
            for i in reversed(range(len(cls))):
                m = cls[i]
                if isinstance(m, nn.Linear):
                    in_f = m.in_features
                    cls[i] = nn.Sequential(
                        nn.Dropout(p=0.8),
                        nn.Linear(in_f, num_classes)
                    )
                    break
                if isinstance(m, nn.Conv2d):
                    cls[i] = nn.Sequential(
                        nn.Dropout2d(p=0.8),
                        nn.Conv2d(m.in_channels, num_classes,
                                  kernel_size=m.kernel_size,
                                  stride=m.stride,
                                  padding=m.padding)
                    )
                    break
            model.classifier = cls
        else:
            in_f = cls.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.8),
                nn.Linear(in_f, num_classes)
            )
    else:
        raise NotImplementedError(f"{name} not Supported!")

    return model

def Net():
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
    ds = configJSON.get("dataset") or configJSON["client_details"][0]["dataset"]
    dataset_name = normalize_dataset_name(ds)
    model_name   = configJSON["client_details"][0]["model"]
    num_classes  = AVAILABLE_DATASETS[dataset_name]["num_classes"]
    return get_dynamic_model(num_classes, model_name)

def get_non_iid_indices(dataset,
                        remove_class_frac,
                        add_class_frac,
                        remove_pct_range,
                        add_pct_range):
    cls2idx = {}
    for i, (_, lbl) in enumerate(dataset):
        cls2idx.setdefault(lbl, []).append(i)

    classes = list(cls2idx.keys())
    n_cls = len(classes)

    n_remove = max(1, int(remove_class_frac * n_cls))
    remove_cls = random.sample(classes, n_remove)

    avail = [c for c in classes if c not in remove_cls]
    raw_add = max(1, int(add_class_frac * n_cls))
    n_add = min(raw_add, len(avail))
    add_cls = random.sample(avail, n_add)

    pct_remove = {c: random.uniform(*remove_pct_range) for c in remove_cls}
    pct_add    = {c: random.uniform(*add_pct_range)    for c in add_cls}

    selected = []
    for c, idxs in cls2idx.items():
        n = len(idxs)
        if c in pct_remove:
            keep = int(n * (1 - pct_remove[c]))
            selected += random.sample(idxs, keep)
        elif c in pct_add:
            add_n = int(n * pct_add[c])
            selected += idxs + random.choices(idxs, k=add_n)
        else:
            selected += idxs

    total = len(dataset)
    if len(selected) > total:
        selected = random.sample(selected, total)
    elif len(selected) < total:
        selected += random.choices(selected, k=total-len(selected))

    zero_cls = random.choice(classes)
    selected = [i for i in selected if dataset[i][1] != zero_cls]

    if len(selected) > total:
        selected = random.sample(selected, total)
    elif len(selected) < total:
        selected += random.choices(selected, k=total-len(selected))

    return selected

def load_data(client_config, GLOBAL_ROUND_COUNTER, dataset_name_override=None):
    global DATASET_NAME, DATASET_TYPE, DATASET_PERSISTANCE

    DATASET_TYPE = client_config.get("data_distribution_type", "").lower()
    DATASET_PERSISTANCE = client_config.get("data_persistance_type", "")
    dataset_name = dataset_name_override or client_config.get("dataset", "")
    DATASET_NAME = normalize_dataset_name(dataset_name)
    log(INFO, f"round number: {GLOBAL_ROUND_COUNTER}, Persist: {DATASET_PERSISTANCE}")

    if DATASET_NAME not in AVAILABLE_DATASETS:
        raise ValueError(f"[ERROR] Dataset '{DATASET_NAME}' non trovato in AVAILABLE_DATASETS.")
    config = AVAILABLE_DATASETS[DATASET_NAME]
    normalize_params = config["normalize"]

    # Dimensioni e batch
    base_size = {
        "CIFAR10": 32, "CIFAR100": 32, "MNIST": 28,
        "FashionMNIST": 28, "KMNIST": 28,
        "ImageNet100": 256, "OXFORDIIITPET": 256
    }[DATASET_NAME]
    model_name = client_config.get("model", "resnet18").lower()
    target_size = 256 if model_name in ["alexnet","vgg11","vgg13","vgg16","vgg19"] else base_size

    # Trasformazioni
    transforms_list = []
    if DATASET_NAME == "ImageNet100":
        transforms_list = [Resize(224), CenterCrop(224)]
        batch_size = 64
    else:
        if target_size != base_size:
            transforms_list.append(Resize((target_size, target_size)))
        batch_size = 64
    transforms_list += [ToTensor(), Normalize(*normalize_params)]
    trf = Compose(transforms_list)

    # Caricamento trainset / testset
    if DATASET_NAME == "ImageNet100":
        DATA_ROOT = Path(__file__).resolve().parent / "data"
        train_path = DATA_ROOT / "imagenet100-preprocessed" / "train"
        test_path  = DATA_ROOT / "imagenet100-preprocessed" / "test"
        if not os.path.isdir(train_path) or not os.path.isdir(test_path):
            raise FileNotFoundError(
                f"Dataset ImageNet100 non trovato in {train_path} e {test_path}"
            )
        trainset = ImageFolder(train_path, transform=trf)
        testset  = ImageFolder(test_path,  transform=trf)
    else:
        cls = config["class"]
        if DATASET_NAME == "OXFORDIIITPET":
            trainset = cls("./data", split="trainval", download=True, transform=trf)
            testset  = cls("./data", split="test",     download=True, transform=trf)
        else:
            trainset = cls("./data", train=True,  download=True, transform=trf)
            testset  = cls("./data", train=False, download=True, transform=trf)

    if DATASET_TYPE == "non-iid":
        classes = list({lbl for _, lbl in trainset})
        n_cls = len(classes)
        remove_frac = random.uniform(1/n_cls, (n_cls-1)/n_cls)
        add_frac    = random.uniform(0,     (n_cls-1)/n_cls)
        low_r, high_r = sorted((random.uniform(0.5,1.0), random.uniform(0.5,1.0)))
        low_a, high_a = sorted((random.uniform(0.5,1.0), random.uniform(0.5,1.0)))

        idxs = get_non_iid_indices(
            trainset,
            remove_frac,
            add_frac,
            (low_r, high_r),
            (low_a, high_a),
        )
        base = Subset(trainset, idxs)

        if HETEROGENEOUS_DATA_HANDLER:
            trainset = balance_dataset_with_gan(
                base,
                num_classes=config["num_classes"],
                target_per_class=len(base) // config["num_classes"],
            )
            ds_name = client_config.get("dataset", "").lower()
            if "cifar" in ds_name:
                max_limit = 5000
            elif "imagenet" in ds_name:
                max_limit = 1300
            else:
                max_limit = len(base) // config["num_classes"]

            trainset = truncate_dataset(trainset, max_limit)
        else:
            trainset = base

    config_path = os.path.join(os.path.dirname(__file__), 'configuration', 'config.json')
    with open(config_path, 'r') as f:
        total_rounds = json.load(f).get("rounds")

    # carica tutto (default)
    if DATASET_PERSISTANCE == "Same Data":   
        pass
    else:
        class_to_indices = defaultdict(list)
        for idx in range(len(trainset)):
            _, label = trainset[idx]
            class_to_indices[label].append(idx)

        selected_indices = []

        for label, indices in class_to_indices.items():
            n_total = len(indices)
            if DATASET_PERSISTANCE == "New Data":
                n_take = int(n_total * GLOBAL_ROUND_COUNTER / total_rounds)
            elif DATASET_PERSISTANCE == "Remove Data":
                n_take = int(n_total * (total_rounds - GLOBAL_ROUND_COUNTER + 1) / total_rounds)
            else:
                n_take = n_total

            if n_take > 0:
                selected_indices.extend(indices[:n_take])

        trainset = Subset(trainset, selected_indices)

    trainloader = DataLoader(TensorLabelDataset(trainset), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def truncate_dataset(dataset, max_per_class: int):
    counts = defaultdict(int)
    kept_indices = []
    for idx, (_, lbl) in enumerate(dataset):
        lbl = int(lbl)  
        if counts[lbl] < max_per_class:
            kept_indices.append(idx)
            counts[lbl] += 1
    return Subset(dataset, kept_indices)

def balance_dataset_with_gan(
    trainset,
    num_classes,
    target_per_class=None,
    latent_dim=100,
    epochs=1,
    batch_size=32,
    device=DEVICE,
):
    counts = Counter(lbl for _, lbl in trainset)
    total = len(trainset)
    if target_per_class is None:
        target_per_class = total // num_classes

    under_cls = [c for c, cnt in counts.items() if 0 < cnt < target_per_class]
    if not under_cls:
        return trainset

    idxs = [i for i, (_, lbl) in enumerate(trainset) if lbl in under_cls]

    C, H, W = trainset[0][0].shape
    target_size = get_valid_downscale_size(min(H, W))  
    if H != target_size or W != target_size:
        resize_for_gan = Compose([
            ToPILImage(),
            Resize((target_size, target_size)),
            ToTensor(),
        ])
        train_for_gan = [(resize_for_gan(img), lbl) for img, lbl in trainset]
    else:
        train_for_gan = list(trainset)

    # Log e subset per GAN
    log(INFO, f"[HDH GAN] Applying GAN to rebalance classes: {under_cls}")
    subset = Subset(train_for_gan, idxs)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    import torchvision
    _orig_make_grid = torchvision.utils.make_grid
    def _make_grid_wrapper(*args, **kwargs):
        if 'range' in kwargs:
            kwargs['value_range'] = kwargs.pop('range')
        return _orig_make_grid(*args, **kwargs)
    torchvision.utils.make_grid = _make_grid_wrapper

    models_cfg = {
        'generator': {'name': DCGANGenerator, 'args': {'encoding_dims': latent_dim, 'out_size': target_size, 'out_channels': C},
                      'optimizer': {'name': torch.optim.Adam, 'args': {'lr': 2e-4, 'betas': (0.5, 0.999)}}},
        'discriminator': {'name': DCGANDiscriminator, 'args': {'in_size': target_size, 'in_channels': C},
                          'optimizer': {'name': torch.optim.Adam, 'args': {'lr': 2e-4, 'betas': (0.5, 0.999)}}},
    }
    losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]

    log(INFO, "[HDH GAN] Starting GAN training...")
    trainer = Trainer(models=models_cfg, losses_list=losses, device=device, sample_size=batch_size, epochs=epochs)
    trainer.train(loader)

    synth_imgs, synth_lbls = [], []
    for c in under_cls:
        cnt = counts[c]
        to_gen = target_per_class - cnt
        if to_gen <= 0:
            continue
        z = torch.randn(to_gen, latent_dim, device=device)
        with torch.no_grad():
            gen = trainer.generator(z).cpu()
        synth_imgs.append(gen)
        synth_lbls += [c] * to_gen

    
    if synth_imgs:
        all_imgs_gan = torch.cat(synth_imgs, dim=0)
        downsample = Compose([ToPILImage(), Resize((H, W)), ToTensor()])
        resized = torch.stack([downsample(img) for img in all_imgs_gan])
        all_lbls = torch.tensor(synth_lbls, dtype=torch.long)
        synth_ds = TensorDataset(resized, all_lbls)
        result = ConcatDataset([trainset, synth_ds])
        log(INFO, f"[HDH GAN] GAN Training Completed.")
        log(INFO, f"[HDH GAN] Rebalanced dataset size: {len(result)} (added {len(synth_lbls)} samples)")
        return result

    return trainset

def train(net, trainloader, valloader, epochs, DEVICE):
    labels = [lbl.item() if isinstance(lbl, torch.Tensor) else lbl for _, lbl in trainloader.dataset]
    dist = dict(Counter(labels))
    log(INFO, f"Training dataset distribution ({DATASET_NAME}): {dist}")

    log(INFO, f"Starting training...")
    start_time = time.time()
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")
    global TRAIN_COMPLETED_TS
    TRAIN_COMPLETED_TS = start_time + training_time
    train_loss, train_acc, train_f1, train_mae = test(net, trainloader)
    val_loss, val_acc, val_f1, val_mae = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "train_mae": train_mae,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_mae": val_mae,
    }
    return results, training_time

def test(net, loader):
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        mae = np.mean(np.abs(np.array(all_labels) - np.array(all_preds)))
    except:
        mae = None
    return avg_loss, accuracy, f1, mae

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    state_dict._metadata = {"": {"version": 2}} 
    net.load_state_dict(state_dict, strict=True)
