import os
import copy
import logging
from dataclasses import dataclass
import pyrallis
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from transformers import ConvNextV2Config, ConvNextV2Model
from sklearn.metrics import f1_score

from src.models import ConvNextV2Classifier


@dataclass
class Config:
    data_dir: str = "data/brodatz_exp"
    batch_size: int = 32
    image_size: int = 224
    epochs: int = 25
    lr: float = 1e-4
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: str = 'ConvNextV2Model'


@pyrallis.wrap()
def main(config: Config):
    logging.basicConfig(filename=f'logs/{config.model}.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(
            config.image_size,
            scale=(0.8, 1.0)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(config.data_dir, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(config.data_dir, "val"),
        transform=test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(config.data_dir, "test"),
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    NUM_CLASSES = len(train_dataset.classes)

    if config.model == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    if config.model == "ConvNextV2Model":
        convnext_config = ConvNextV2Config(
            num_channels=3,
            num_labels=NUM_CLASSES
        )
        backbone = ConvNextV2Model(convnext_config)
        model = ConvNextV2Classifier(backbone, NUM_CLASSES)

    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )

    best_acc = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        model.train()
        running_loss = 0
        running_correct = 0
        train_preds = []
        train_labels = []
        for images, labels in train_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = running_correct / len(train_dataset)
        train_f1 = f1_score(
            train_labels,
            train_preds,
            average="macro"
        )

        model.eval()
        running_loss = 0
        running_correct = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.device)
                labels = labels.to(config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                running_loss += loss.item() * images.size(0)
                running_correct += (preds == labels).sum().item()

        val_loss = running_loss / len(val_dataset)
        val_f1 = f1_score(
            val_labels,
            val_preds,
            average="macro"
        )
        val_acc = running_correct / len(val_dataset)

        scheduler.step()

        logging.info(f"Epoch: {epoch}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Train Acc : {train_acc:.4f}")
        logging.info(f"Train F1  : {train_f1:.4f}")

        logging.info(f"Val Loss  : {val_loss:.4f}")
        logging.info(f"Val Acc   : {val_acc:.4f}")
        logging.info(f"Val F1    : {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    correct = 0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            outputs = model(images)
            preds = outputs.argmax(1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()

    test_acc = correct / len(test_dataset)
    test_f1 = f1_score(
        test_labels,
        test_preds,
        average="macro"
    )

    logging.info(f"\nBest validation accuracy: {best_acc:.4f}")
    logging.info(f"Test accuracy: {test_acc:.4f}")
    logging.info(f"Test F1-score: {test_f1:.4f}")

    torch.save(model.state_dict(), f"models/brodatz_{config.model}.pth")


if __name__ == "__main__":
    main()
