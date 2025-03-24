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
from tqdm import tqdm  # Importo tqdm per la barra di avanzamento

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1 
if torch.cuda.is_available():
    print("Using CUDA")
else:
    print("Using CPU")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
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
        self.fc2 = nn.Linear(1024, 100)  # 100 classi per ImageNet-100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.avg_pool(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_data(data_path=None, batch_size=16):
    """Carica il dataset ImageNet-100 (set di training e validazione).
    
    Se non viene specificato data_path, verrà cercata la cartella "imagenet100"
    nello stesso livello di questo file. La struttura attesa è:
      - imagenet100/
          - train/   -> 1300 immagini per classe, organizzate in sottocartelle
          - val/     -> 50 immagini per classe, organizzate in sottocartelle
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet100")
    
    # Trasformazioni per training e validazione
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
    
    # Costruisci i path per le cartelle di training e validazione
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    
    # Crea i dataset
    trainset = ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
    valset = ImageFolder(
        root=val_dir,
        transform=val_transform
    )
    
    # Crea i data loader
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Regola in base al tuo sistema
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
    
    # Scheduler per il learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # Uso tqdm per mostrare il progresso dei batch in ogni epoca
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        log(INFO, f"Epoch {epoch+1}/{epochs} completed. Average Loss: {running_loss/len(trainloader):.4f}")

    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")

    # Valutazione su training e validazione
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
    """Valuta il modello sul set di test."""
    net.to(DEVICE)
    net.eval()
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

    accuracy = correct / len(testloader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score_torch(all_labels, all_preds, num_classes=100, average='macro')
    
    return total_loss / len(testloader), accuracy, f1

def f1_score_torch(y_true, y_pred, num_classes, average='macro'):
    """Calcola l'F1 score utilizzando operazioni di PyTorch."""
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_true.device)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t.long(), p.long()] += 1

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
    """Restituisce i pesi del modello come lista di array NumPy."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Imposta i pesi del modello da una lista di array NumPy."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Dizionario di mapping delle classi per ImageNet-100
IMAGENET100_CLASSES = {
    "n01968897": "chambered nautilus, pearly nautilus, nautilus",
    "n01770081": "harvestman, daddy longlegs, Phalangium opilio",
    "n01818515": "macaw",
    "n02011460": "bittern",
    "n01496331": "electric ray, crampfish, numbfish, torpedo",
    "n01847000": "drake",
    "n01687978": "agama",
    "n01740131": "night snake, Hypsiglena torquata",
    "n01537544": "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "n01491361": "tiger shark, Galeocerdo cuvieri",
    "n02007558": "flamingo",
    "n01735189": "garter snake, grass snake",
    "n01630670": "common newt, Triturus vulgaris",
    "n01440764": "tench, Tinca tinca",
    "n01819313": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "n02002556": "white stork, Ciconia ciconia",
    "n01667778": "terrapin",
    "n01755581": "diamondback, diamondback rattlesnake, Crotalus adamanteus",
    "n01924916": "flatworm, platyhelminth",
    "n01751748": "sea snake",
    "n01984695": "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "n01729977": "green snake, grass snake",
    "n01614925": "bald eagle, American eagle, Haliaeetus leucocephalus",
    "n01608432": "kite",
    "n01443537": "goldfish, Carassius auratus",
    "n01770393": "scorpion",
    "n01855672": "goose",
    "n01560419": "bulbul",
    "n01592084": "chickadee",
    "n01914609": "sea anemone, anemone",
    "n01582220": "magpie",
    "n01667114": "mud turtle",
    "n01985128": "crayfish, crawfish, crawdad, crawdaddy",
    "n01820546": "lorikeet",
    "n01773797": "garden spider, Aranea diademata",
    "n02006656": "spoonbill",
    "n01986214": "hermit crab",
    "n01484850": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "n01749939": "green mamba",
    "n01828970": "bee eater",
    "n02018795": "bustard",
    "n01695060": "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "n01729322": "hognose snake, puff adder, sand viper",
    "n01677366": "common iguana, iguana, Iguana iguana",
    "n01734418": "king snake, kingsnake",
    "n01843383": "toucan",
    "n01806143": "peacock",
    "n01773549": "barn spider, Araneus cavaticus",
    "n01775062": "wolf spider, hunting spider",
    "n01728572": "thunder snake, worm snake, Carphophis amoenus",
    "n01601694": "water ouzel, dipper",
    "n01978287": "Dungeness crab, Cancer magister",
    "n01930112": "nematode, nematode worm, roundworm",
    "n01739381": "vine snake",
    "n01883070": "wombat",
    "n01774384": "black widow, Latrodectus mactans",
    "n02037110": "oystercatcher, oyster catcher",
    "n01795545": "black grouse",
    "n02027492": "red-backed sandpiper, dunlin, Erolia alpina",
    "n01531178": "goldfinch, Carduelis carduelis",
    "n01944390": "snail",
    "n01494475": "hammerhead, hammerhead shark",
    "n01632458": "spotted salamander, Ambystoma maculatum",
    "n01698640": "American alligator, Alligator mississipiensis",
    "n01675722": "banded gecko",
    "n01877812": "wallaby, brush kangaroo",
    "n01622779": "great grey owl, great gray owl, Strix nebulosa",
    "n01910747": "jellyfish",
    "n01860187": "black swan, Cygnus atratus",
    "n01796340": "ptarmigan",
    "n01833805": "hummingbird",
    "n01685808": "whiptail, whiptail lizard",
    "n01756291": "sidewinder, horned rattlesnake, Crotalus cerastes",
    "n01514859": "hen",
    "n01753488": "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "n02058221": "albatross, mollymawk",
    "n01632777": "axolotl, mud puppy, Ambystoma mexicanum",
    "n01644900": "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "n02018207": "American coot, marsh hen, mud hen, water hen, Fulica americana",
    "n01664065": "loggerhead, loggerhead turtle, Caretta caretta",
    "n02028035": "redshank, Tringa totanus",
    "n02012849": "crane",
    "n01776313": "tick",
    "n02077923": "sea lion",
    "n01774750": "tarantula",
    "n01742172": "boa constrictor, Constrictor constrictor",
    "n01943899": "conch",
    "n01798484": "prairie chicken, prairie grouse, prairie fowl",
    "n02051845": "pelican",
    "n01824575": "coucal",
    "n02013706": "limpkin, Aramus pictus",
    "n01955084": "chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "n01773157": "black and gold garden spider, Argiope aurantia",
    "n01665541": "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
    "n01498041": "stingray",
    "n01978455": "rock crab, Cancer irroratus",
    "n01693334": "green lizard, Lacerta viridis",
    "n01950731": "sea slug, nudibranch",
    "n01829413": "hornbill",
    "n01514668": "cock"
}
