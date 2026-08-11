# Эксперименты по федеративному обучению

Код для федеративного обучения классификатора текстур (FastViT-T8 + Flower FedAvg).

## Подготовка данных

Исходные изображения лежат в папках-классах (clear, G, GP, M, T).
Скрипт конвертирует их в grayscale и делит на train/val/test:

```shell
python -m src.datasets.preprocessing
```

Пути по умолчанию: `data/dataset2_exp` -> `data/dataset2_exp_prepared`
(совпадают с `dataset.root` в конфиге).

## Запуск

### Локальная симуляция (один процесс, Ray)

```shell
python main.py --mode simulation
```

### Распределённый запуск (server + clients)

```shell
# терминал 1
python main.py --mode server

# терминал 2+
python main.py --mode client --client-id 0
python main.py --mode client --client-id 1
```

Конфигурация: `configs/texture.yaml`.
