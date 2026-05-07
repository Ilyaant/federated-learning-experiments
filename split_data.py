import numpy as np
import torchvision
from torch.utils.data import Subset
from collections import Counter


def get_shared_and_remaining_indices(targets, shared_ratio=0.1):
    """
    Выделяет общую выборку (shared_ratio от датасета) с сохранением баланса классов.
    Возвращает индексы общей выборки и индексы оставшихся данных.
    """
    shared_indices = []
    remaining_indices = []
    
    # Группируем индексы по классам (от 0 до 99)
    class_indices = {i: [] for i in range(100)}
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)
        
    for cls, indices in class_indices.items():
        np.random.shuffle(indices)
        split_point = int(len(indices) * shared_ratio)
        
        shared_indices.extend(indices[:split_point])
        remaining_indices.extend(indices[split_point:])
        
    return shared_indices, remaining_indices, class_indices


def split_cifar100(dataset, num_clients=5, strategy="iid", shared_ratio=0.1, **kwargs):
    """
    Разбивает датасет на клиентов согласно стратегии + добавляет каждому общую выборку.
    """
    targets = np.array(dataset.targets)
    
    # 1. Выделяем общую выборку
    shared_idx, remaining_idx, class_indices = get_shared_and_remaining_indices(targets, shared_ratio)
    np.random.shuffle(remaining_idx)
    
    client_local_indices = {i: [] for i in range(num_clients)}

    # 2. Распределяем оставшиеся данные по стратегиям
    if strategy == "iid":
        # Стратегия 1: Равномерное распределение
        splits = np.array_split(remaining_idx, num_clients)
        for i in range(num_clients):
            client_local_indices[i] = splits[i].tolist()
            
    elif strategy == "quantity_skew":
        # Стратегия 2: Разный объем (Quantity Skew)
        # По умолчанию: 10%, 15%, 20%, 25%, 30% от оставшихся данных
        proportions = kwargs.get('proportions', [0.1, 0.15, 0.2, 0.25, 0.3])
        assert sum(proportions) == 1.0, "Сумма пропорций должна быть равна 1"
        
        split_points = (np.cumsum(proportions) * len(remaining_idx)).astype(int)[:-1]
        splits = np.split(remaining_idx, split_points)
        for i in range(num_clients):
            client_local_indices[i] = splits[i].tolist()
            
    elif strategy == "dirichlet":
        # Стратегия 3: Дисбаланс распределения Дирихле (Label Distribution Skew)
        alpha = kwargs.get('alpha', 0.5)
        
        # Пересобираем оставшиеся индексы по классам
        rem_class_indices = {i: [] for i in range(100)}
        for idx in remaining_idx:
            rem_class_indices[targets[idx]].append(idx)
            
        for cls in range(100):
            # Генерируем пропорции для 5 клиентов из распределения Дирихле
            proportions = np.random.dirichlet([alpha] * num_clients)
            
            cls_indices = rem_class_indices[cls]
            np.random.shuffle(cls_indices)
            
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            splits = np.split(cls_indices, split_points)
            
            for i in range(num_clients):
                client_local_indices[i].extend(splits[i].tolist())
                
    elif strategy == "pathological":
        # Стратегия 4: Патологический Non-IID (Клиент видит только уникальные классы)
        # 100 классов / 5 клиентов = по 20 уникальных классов на клиента
        rem_class_indices = {i: [] for i in range(100)}
        for idx in remaining_idx:
            rem_class_indices[targets[idx]].append(idx)
            
        classes_per_client = 100 // num_clients
        classes_assigned = np.random.permutation(100) # Перемешиваем классы
        
        for i in range(num_clients):
            client_classes = classes_assigned[i*classes_per_client : (i+1)*classes_per_client]
            for cls in client_classes:
                client_local_indices[i].extend(rem_class_indices[cls])
                
    else:
        raise ValueError("Неизвестная стратегия")

    # 3. Объединяем локальные данные с общей выборкой для каждого клиента
    client_final_datasets = []
    for i in range(num_clients):
        # Перемешиваем, чтобы общая выборка не лежала куском в начале/конце
        combined_indices = client_local_indices[i] + shared_idx
        np.random.shuffle(combined_indices)
        
        # Создаем Subset PyTorch для удобства дальнейшего использования в DataLoader
        # client_final_datasets[i] = Subset(dataset, combined_indices)
        client_final_datasets.append(Subset(dataset, combined_indices))
        
    return client_final_datasets, client_local_indices, shared_idx


# --- ДЕМОНСТРАЦИЯ РАБОТЫ И АНАЛИЗ ---
def analyze_split(client_local_indices, shared_idx, original_dataset, strategy_name):
    print(f"\n{'='*60}")
    print(f"СТРАТЕГИЯ: {strategy_name.upper()}")
    
    # Анализируем общую выборку один раз
    shared_targets = [original_dataset.targets[idx] for idx in shared_idx]
    shared_unique_classes = len(set(shared_targets))
    print(f"ОБЩАЯ ВЫБОРКА (Shared): {len(shared_idx)} фото, Уникальных классов: {shared_unique_classes}/100")
    print(f"{'-'*60}")
    
    for client_id, local_indices in client_local_indices.items():
        # Считаем метки ТОЛЬКО в индивидуальной (локальной) части
        local_targets = [original_dataset.targets[idx] for idx in local_indices]
        local_unique_classes = len(set(local_targets))
        
        # Считаем метки в смешанном датасете (как будет видеть нейросеть)
        combined_unique_classes = len(set(local_targets + shared_targets))
        
        print(f"Клиент {client_id+1}:")
        print(f"  -- Локальная часть: {len(local_targets)} фото | Уникальных классов: {local_unique_classes}/100")
        
        # Дополнительно: покажем дисбаланс (Топ-3 самых частых класса в локальной выборке)
        if len(local_targets) > 0:
            top_3_classes = Counter(local_targets).most_common(3)
            top_3_str = ", ".join([f"Класс {c}: {cnt} шт." for c, cnt in top_3_classes])
            print(f"     Самые частые локальные классы: {top_3_str}")
            
        print(f"  -- ИТОГО ДЛЯ ОБУЧЕНИЯ (Лок. + Общ.): {len(local_targets) + len(shared_idx)} фото | Уникальных классов: {combined_unique_classes}/100\n")


if __name__ == "__main__":
    np.random.seed(42)
    
    # Чтобы не скачивать каждый раз, download=False (если уже скачан)
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    
    shared_ratio = 0.10
    strategies = ["iid", "quantity_skew", "dirichlet", "pathological"]
    
    for strat in strategies:
        kwargs = {'alpha': 0.1} if strat == "dirichlet" else {} # alpha=0.1 для жесткого дисбаланса
        
        # Теперь получаем 3 переменные: готовые датасеты, локальные индексы, общие индексы
        client_datasets, local_indices, shared_idx = split_cifar100(
            dataset=cifar100_train, 
            num_clients=5, 
            strategy=strat, 
            shared_ratio=shared_ratio,
            **kwargs
        )
        
        # Анализатор теперь смотрит внутрь "кухни"
        analyze_split(local_indices, shared_idx, cifar100_train, strat)
