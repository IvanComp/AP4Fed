import os
import re
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models as tv_models
from torchcam.methods import SmoothGradCAMpp
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
from cycler import cycler
import numpy as np

cmap = mpl.cm.RdYlGn
mpl.rcParams['image.cmap'] = 'RdYlGn'
mpl.rcParams['axes.prop_cycle'] = cycler('color', cmap(np.linspace(0, 1, 10)))

CIFAR10_CLASSES = [
    'aereo', 'automobile', 'uccello', 'gatto', 'cervo',
    'cane', 'rana', 'cavallo', 'nave', 'camion'
]
MNIST_CLASSES = [str(i) for i in range(10)]

# Configuration
SERVER_DIR = "./model_weights/server"
CLIENTS_DIR = "./model_weights/clients"
OUTPUT_DIR = "./output"
NUM_IMAGES = 5
AVAILABLE_DATASETS = ["MNIST", "CIFAR10", "ImageNet100"]

# Custom CNN definitions
class CNN16k(nn.Module):
    def __init__(self, input_size, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        dummy = torch.zeros(1, in_ch, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flat = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat, 60)
        self.fc2 = nn.Linear(60, 42)
        self.fc3 = nn.Linear(42, 10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CNN64k(nn.Module):
    def __init__(self, input_size, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        dummy = torch.zeros(1, in_ch, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flat = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CNN256k(nn.Module):
    def __init__(self, input_size, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        dummy = torch.zeros(1, in_ch, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flat = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Model selection
TORCHVISION_MODELS = [
    name for name in dir(tv_models)
    if not name.startswith("__") and callable(getattr(tv_models, name))
]
CUSTOM_MODELS = {"CNN16k": CNN16k, "CNN64k": CNN64k, "CNN256k": CNN256k}
ALL_MODELS = list(CUSTOM_MODELS.keys()) + TORCHVISION_MODELS

# Dataset selection
def select_dataset():
    print("Select dataset:")
    for i, name in enumerate(AVAILABLE_DATASETS, 1): print(f" {i}. {name}")
    choice = 0
    while not 1 <= choice <= len(AVAILABLE_DATASETS):
        try: choice = int(input(f"Enter choice [1-{len(AVAILABLE_DATASETS)}]: "))
        except: pass
    return AVAILABLE_DATASETS[choice-1]

# Model selector uses chosen dataset to set input size and channels
def select_model(dataset_name):
    print("Select model to evaluate:")
    for i, name in enumerate(ALL_MODELS, 1): print(f" {i}. {name}")
    choice = 0
    while not 1 <= choice <= len(ALL_MODELS):
        try: choice = int(input(f"Enter choice [1-{len(ALL_MODELS)}]: "))
        except: pass
    name = ALL_MODELS[choice-1]
    if name in CUSTOM_MODELS:
        size_map = {"MNIST":28, "CIFAR10":32, "ImageNet100":224}
        ch_map = {"MNIST":1, "CIFAR10":3, "ImageNet100":3}
        return CUSTOM_MODELS[name](input_size=size_map[dataset_name], in_ch=ch_map[dataset_name])
    else:
        return getattr(tv_models, name)(pretrained=False)

# SSIM computation
def compute_state_dict_ssim(sd_a, sd_b):
    total, count = 0.0, 0
    for key in sd_a:
        a = sd_a[key].cpu().numpy(); b = sd_b[key].cpu().numpy()
        if a.shape != b.shape: continue
        amin, amax = a.min(), a.max(); bmin, bmax = b.min(), b.max()
        eps=1e-8
        an=(a-amin)/(amax-amin+eps); bn=(b-bmin)/(bmax-bmin+eps)
        if an.ndim==1:
            an=an[np.newaxis,:]; bn=bn[np.newaxis,:]
        else:
            an=an.reshape(an.shape[0],-1); bn=bn.reshape(bn.shape[0],-1)
        for x,y in zip(an,bn):
            L=x.shape[0]; w=min(7,L)
            if w%2==0: w=max(1,w-1)
            total+=ssim(x,y,data_range=1.0,win_size=w,channel_axis=None); count+=1
    return total/count if count else 0

# Load models
def load_models(sdir, cdir):
    rounds = sorted(
        int(m.group(1)) for f in os.listdir(sdir)
        for m in [re.match(r"MW_round(\d+)\.pt", f)] if m
    )
    out={}
    for r in rounds:
        g= torch.load(os.path.join(sdir,f"MW_round{r}.pt"), map_location="cpu")
        gsd=g.state_dict() if isinstance(g,torch.nn.Module) else g
        cds={}
        for cid in os.listdir(cdir):
            fpath=os.path.join(cdir,cid,f"MW_round{r}.pt")
            if os.path.isfile(fpath):
                c= torch.load(fpath, map_location="cpu")
                cds[int(cid)] = c.state_dict() if isinstance(c,torch.nn.Module) else c
        out[r] = {"global":gsd, "clients":cds}
    return out

# Plotting utilities
def heatmap(mat, labels, title, path):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect='auto', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_series(data,path):
    plt.figure(figsize=(8,6))
    for cid,series in data.items(): xs=sorted(series); ys=[series[x] for x in xs]; plt.plot(xs,ys,marker='o',label=str(cid))
    plt.legend(); plt.xlabel('Round'); plt.ylabel('SSIM'); plt.tight_layout(); plt.savefig(path); plt.close()

def visualize(latest_sd, dataset_name, n, path):
    # 1) Dataset + classi
    if dataset_name == 'MNIST':
        ds = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
        classes = MNIST_CLASSES
    elif dataset_name == 'CIFAR10':
        ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
        classes = CIFAR10_CLASSES
    else:  # ImageNet100
        ds = datasets.ImageFolder(
            './data/imagenet100-preprocessed/test',
            transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        )
        classes = ds.classes

    # 2) Modello + pesi
    model = select_model(dataset_name).to('cpu')
    model.load_state_dict(latest_sd)
    model.eval()

    # 3) Trova l'ultimo Conv2d per la Grad-CAM
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise RuntimeError("Impossibile trovare layer Conv2d per Grad-CAM")
    target_layer = conv_layers[-1]
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)

    # 4) Campiona n esempi
    idxs = random.sample(range(len(ds)), n)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))

    for i, idx in enumerate(idxs):
        img, lbl = ds[idx]
        batch = img.unsqueeze(0)

        # forward (senza no_grad)
        out  = model(batch)
        pred = out.argmax(1).item()

        # ————— qui il fix Grad-CAM —————
        raw_cam = cam_extractor(pred, out)
        # Se è dict, prendi la prima mappa; se è lista, prendi l’elemento 0
        if isinstance(raw_cam, dict):
            cam_map = next(iter(raw_cam.values()))
        elif isinstance(raw_cam, list):
            cam_map = raw_cam[0]
        else:
            cam_map = raw_cam
        cam_map = cam_map.squeeze().cpu().numpy()
        # Colonna 1: immagine + label vera
        ax = axes[i,0]; ax.axis('off')
        arr = img.squeeze().permute(1,2,0).numpy() if img.ndim==3 else img.squeeze().numpy()
        ax.imshow(arr, cmap=None if img.ndim==3 else 'gray')
        ax.set_title(f'Vera: {classes[lbl]}')

        # Colonna 2: Grad-CAM sovrapposta
        ax = axes[i,1]; ax.axis('off')
        ax.imshow(arr, alpha=0.5, cmap=None if img.ndim==3 else 'gray')
        ax.imshow(cam_map, cmap='jet', alpha=0.5)
        ax.set_title('Grad-CAM')

        # Colonna 3: predizione
        ax = axes[i,2]; ax.axis('off')
        ax.text(0.5,0.5,f'Pred: {classes[pred]}',
                ha='center',va='center',fontsize=14)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

# Main
if __name__=='__main__':
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    dataset_name = select_dataset()
    models = load_models(SERVER_DIR,CLIENTS_DIR)
    scores = {}
    for r,d in models.items():
        gs = d['global']; scores[r] = {cid: compute_state_dict_ssim(gs, sd) for cid,sd in d['clients'].items()}
        labs = sorted(scores[r]); mat = np.array([scores[r][c] for c in labs])[None,:]
        heatmap(mat, [f'C{c}' for c in labs], f'R{r}', os.path.join(OUTPUT_DIR,f'hm_{r}.png'))
    ts={}
    for r,sc in scores.items():
        for cid,v in sc.items(): ts.setdefault(cid,{})[r]=v
    plot_series(ts, os.path.join(OUTPUT_DIR,'series.png'))
    latest = max(models); visualize(models[latest]['global'], dataset_name, NUM_IMAGES, os.path.join(OUTPUT_DIR,'dataset_vis.png'))
    print('Done', OUTPUT_DIR)
