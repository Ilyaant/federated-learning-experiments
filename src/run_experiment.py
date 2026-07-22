import os
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from torch.utils.data import DataLoader

# ==========================
# Настройки
# ==========================

DATA_DIR = "data/brodatz_exp"

BATCH_SIZE = 32
IMAGE_SIZE = 224

EPOCHS = 25

LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(filename='logs/efficientnet_v2_s.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# ==========================
# Аугментации
# ==========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================
# Dataset
# ==========================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

NUM_CLASSES = len(train_dataset.classes)

print(train_dataset.classes)

# ==========================
# Модель
# ==========================

weights = EfficientNet_V2_S_Weights.DEFAULT

model = efficientnet_v2_s(weights=weights)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

# ==========================
# Loss
# ==========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ==========================
# Train
# ==========================

best_acc = 0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    ##########################
    # TRAIN
    ##########################

    model.train()

    running_loss = 0
    running_correct = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        preds = outputs.argmax(1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = running_correct / len(train_dataset)

    ##########################
    # VALIDATION
    ##########################

    model.eval()

    running_loss = 0
    running_correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            preds = outputs.argmax(1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()

    val_loss = running_loss / len(val_dataset)
    val_acc = running_correct / len(val_dataset)

    scheduler.step()

    logging.info(f"Train Loss: {train_loss:.4f}")
    logging.info(f"Train Acc : {train_acc:.4f}")

    logging.info(f"Val Loss  : {val_loss:.4f}")
    logging.info(f"Val Acc   : {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())

# ==========================
# TEST
# ==========================

model.load_state_dict(best_weights)

model.eval()

correct = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()

test_acc = correct / len(test_dataset)

logging.info(f"\nBest validation accuracy: {best_acc:.4f}")
logging.info(f"Test accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "models/brodatz_efficientnetv2.pth")