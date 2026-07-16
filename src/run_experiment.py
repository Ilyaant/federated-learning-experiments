import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchvision
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict
import numpy as np
import random
import logging

from src.split_data import split_cifar100, analyze_split
from src.models import CIFARMediumCNN, train, test
from src.metrics import print_client_distribution_metrics, weighted_average


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class CIFARClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, epochs, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.device = device

    def get_parameters(self):
        return[val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=self.epochs, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters):
        self.set_parameters(parameters)
        loss, accuracy, precision, recall, f1 = test(self.model, self.testloader, device=self.device)

        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }


def run_experiment(
    split_strategy: str,
    shared_ratio: float,
    num_clients: int = 5,
    num_rounds: int = 5,
    epochs: int = 5,
    device: str = 'cpu',
    model_name: str = 'CIFARMediumCNN',
    num_classes: int = 100,
    batch_size: int = 64,
    metrics_log_file: str = '../logs/metrics_0.txt'
):
    logging.basicConfig(filename=metrics_log_file,
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
    logging.info(f"Эксперимент с разбиением: {split_strategy}")
    # Подготовка данных
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((32, 32)),
        #v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # from ImageNet
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)

    kwargs = {'alpha': 0.1} if split_strategy == "dirichlet" else {}
    client_datasets, local_indices, shared_idx = split_cifar100(
        dataset=cifar100_train, 
        num_clients=num_clients, 
        strategy=split_strategy, 
        shared_ratio=shared_ratio,
        **kwargs
    )
    analyze_split(local_indices, shared_idx, cifar100_train, split_strategy)
    cifar100_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=test_transform)
    print_client_distribution_metrics(client_datasets, num_classes=num_classes)
    testloader = DataLoader(cifar100_test, batch_size=batch_size)

    # Функция для генерации клиента по его ID
    def client_fn(cid: str) -> fl.client.Client:
        if model_name == 'CIFARMediumCNN':
            model = CIFARMediumCNN()
        
        if model_name == 'efficientnet_v2_s':
            model = torchvision.models.efficientnet_v2_s(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        if model_name == 'resnet18':
            model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()

        model = model.to(device)
        trainloader = DataLoader(client_datasets[int(cid)], batch_size=batch_size, shuffle=True)
        return CIFARClient(model, trainloader, testloader, epochs, device).to_client()

    # Стратегия сервера (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Обучаем всех клиентов
        fraction_evaluate=1.0, # Оцениваем на всех клиентах
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average, # Усреднение точности
    )

    # Запуск симуляции (Flower поднимает виртуальные клиенты с использованием Ray)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    return history
