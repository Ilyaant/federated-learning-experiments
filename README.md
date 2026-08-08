# Эксперименты по федеративному обучению

Код для федеративного обучения классификатора текстур (FastViT-T8 + Flower FedAvg).

## Подготовка данных

```shell
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from datasets.preprocessing import prepare_dataset
prepare_dataset('data/raw', 'data/texture')
"
```

## Запуск

```shell
# терминал 1
python main.py --mode server

# терминал 2+
python main.py --mode client --client-id 0
python main.py --mode client --client-id 1
```

Конфигурация: `configs/texture.yaml`.
