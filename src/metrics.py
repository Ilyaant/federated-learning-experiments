import numpy as np


def _entropy(p: np.ndarray) -> float:
    """Энтропия распределения p (p уже нормализован, сумма = 1)."""
    p_safe = p[p > 0]  # убираем нули, чтобы не было log(0)
    return float(-(p_safe * np.log2(p_safe)).sum())


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) c защитой от нулей."""
    p_safe = p + eps
    q_safe = q + eps
    return float((p_safe * (np.log2(p_safe) - np.log2(q_safe))).sum())


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = 0.5*(p+q)."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def print_client_distribution_metrics(client_datasets, num_classes: int = 100):
    """Считает распределение классов по клиентам,
    их энтропию и JS‑дивергенцию относительно глобального распределения.
    """
    num_clients = len(client_datasets)

    # Считаем глобальные частоты классов (по всем клиентам)
    global_counts = np.zeros(num_classes, dtype=np.int64)
    client_counts = []

    for subset in client_datasets:
        # subset — это torch.utils.data.Subset над приватным датасетом
        labels = []
        for idx in subset.indices:
            _, y = subset.dataset[idx]
            labels.append(int(y))
        counts = np.bincount(labels, minlength=num_classes)
        client_counts.append(counts)
        global_counts += counts

    total_samples = int(global_counts.sum())
    global_p = global_counts / total_samples

    js_values = []
    entropies = []
    client_sizes = []

    for i, counts in enumerate(client_counts):
        n_i = int(counts.sum())
        client_sizes.append(n_i)

        if n_i == 0:
            js = 0.0
            H = 0.0
        else:
            p_i = counts / n_i
            js = jensen_shannon_divergence(p_i, global_p)
            H = _entropy(p_i)

        js_values.append(js)
        entropies.append(H)

        print(f"Клиент {i}: N={n_i}, JS={js:.4f}, H={H:.4f}")

    # Взвешенное среднее JS и энтропии по клиентам
    weights = np.array(client_sizes, dtype=np.float64) / total_samples
    js_avg = float((weights * np.array(js_values)).sum())
    H_avg = float((weights * np.array(entropies)).sum())

    print(f"\n[Label skew] Средний взвешенный JS: {js_avg:.4f}")
    print(f"[Label diversity] Средняя взвешенная энтропия клиента: {H_avg:.4f}")


def weighted_average(metrics):
    total_examples = sum([num_examples for num_examples, _ in metrics])
    aggregated_metrics = {}
    for key in metrics[0][1].keys():
        aggregated_metrics[key] = sum([num_examples * m[key] for num_examples, m in metrics]) / total_examples
        
    return aggregated_metrics
