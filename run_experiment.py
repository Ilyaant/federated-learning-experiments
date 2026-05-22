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
import pickle

from split_data import split_cifar100, analyze_split
from models import CIFARMediumCNN, train, test
from metrics import print_client_distribution_metrics, weighted_average


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

EPOCHS = 3
ROUNDS = 100
SHARED_RATIO = 0.1
# SHARED_RATIO = 0
BATCH_SIZE = 64
NUM_CLIENTS = 15
NUM_CLASSES = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# if SHARED_RATIO == 0:
#     metrics_log_file = 'metrics_0.txt'
# else:
#     metrics_log_file = 'metrics.txt'

metrics_log_file = 'metrics_noniid.txt'

logging.basicConfig(filename=metrics_log_file,
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class CIFARClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return[val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=EPOCHS, device=DEVICE)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, precision, recall, f1 = test(self.model, self.testloader, device=DEVICE)

        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }


def run_experiment(split_strategy: str, shared_ratio: float, num_clients: int = 5, num_rounds: int = 5, device: str = 'cpu'):
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
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    print_client_distribution_metrics(client_datasets, num_classes=NUM_CLASSES)
    testloader = DataLoader(cifar100_test, batch_size=BATCH_SIZE)

    # Функция для генерации клиента по его ID
    def client_fn(cid: str) -> fl.client.Client:
        model = CIFARMediumCNN()

        # model = torchvision.models.efficientnet_v2_s(weights=None)
        # model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

        # model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
        # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # model.maxpool = nn.Identity()

        model = model.to(device)
        trainloader = DataLoader(client_datasets[int(cid)], batch_size=BATCH_SIZE, shuffle=True)
        # .to_client() преобразует NumPyClient в базовый Client
        return CIFARClient(model, trainloader, testloader).to_client()

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


if __name__ == "__main__":
    # for share_strategy in ["dirichlet", "pathological", "quantity_skew", "iid"]:
    for share_strategy in ["pathological"]:
        history_exp2 = run_experiment(
            share_strategy, 
            shared_ratio=SHARED_RATIO,
            num_clients=NUM_CLIENTS,
            num_rounds=ROUNDS,
            device=DEVICE
        )

        print("\n--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ (ПОСЛЕДНИЙ РАУНД) ---")
        print("Accuracy:", history_exp2.metrics_distributed['accuracy'][-1][1])
        print("F1-score:", history_exp2.metrics_distributed['f1'][-1][1])

        with open(f'history_{share_strategy}_{SHARED_RATIO}.pkl', 'wb') as f:
            pickle.dump(history_exp2, f)
        
        history_exp2 = run_experiment(
            share_strategy, 
            shared_ratio=0,
            num_clients=NUM_CLIENTS,
            num_rounds=ROUNDS,
            device=DEVICE
        )

        print("\n--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ (ПОСЛЕДНИЙ РАУНД) ---")
        print("Accuracy:", history_exp2.metrics_distributed['accuracy'][-1][1])
        print("F1-score:", history_exp2.metrics_distributed['f1'][-1][1])

        # with open(f'history_{share_strategy}_{SHARED_RATIO}.pkl', 'wb') as f:
        with open(f'history_{share_strategy}_0.pkl', 'wb') as f:
            pickle.dump(history_exp2, f)
