# Эксперименты по классификации текстур

Код для централизованного обучения классификатора текстур (FastViT-T8)
и подбора гиперпараметров по patch-level F1 на валидации.

## Подготовка данных

Исходные изображения лежат в папках-классах (clear, G, GP, M, T).
Скрипт конвертирует их в grayscale и делит на train/val/test **по группам
кадров одного образца** (`15_11_17(2-1)пп(20)-1-3` и `-2-4` не разъезжаются
по сплитам):

```shell
python -m src.datasets.preprocessing
```

Пути по умолчанию: `data/dataset2_exp` -> `data/dataset2_exp_prepared`.

Даже если на диске уже лежит старый file-level сплит, обучение по умолчанию
пересобирает train/val/test в памяти (`dataset.regroup: true`,
`dataset.split_seed: 42`). Чтобы использовать папки as-is:

```shell
python main.py --override dataset.regroup=false
```

## Запуск одного эксперимента

```shell
python main.py --config configs/texture.yaml
```

Частые переопределения:

```shell
python main.py --epochs 30 --downscale 3 --save-dir logs/run_downscale3
python main.py --override train.lr=3e-5 --override dataset.augmentation.blur_prob=0.0
python main.py --patience 10 --selection-metric val_f1 --run-name geom_ds2
```

Результаты пишутся в уникальный каталог под `logging.save_dir`
(`logs/runs/<run_name>_<timestamp>`), если не задан `--save-dir`:

- `metrics.csv` / `history.json` — метрики по эпохам (без test, пока не `eval_test: every`)
- `summary.json` — чекпоинт, выбранный по val, и **одно** измерение test
- `split_manifest.json` — какие группы кадров попали в какой сплит
- `config.yaml` — итоговый конфиг запуска
- `model_best.pt` — лучший чекпоинт по `evaluation.selection_metric` (по умолчанию patch-level `val_f1`)
- `model_final.pt` — веса после последней эпохи
- `model_swa.pt` — среднее лучших `evaluation.swa_window` чекпоинтов (`swa_mode: best`)

Обучение останавливается, если selection-метрика не растёт
`evaluation.early_stopping_patience` эпох. Test считается один раз в конце
на best/SWA, а не каждую эпоху.

## Подбор гиперпараметров

Список джобов: `configs/sweep.yaml` (сейчас — регуляризация поверх победителя
`ds1_geom`: `epoch_fraction`, dropout/weight_decay, меньший lr backbone).

```shell
python main.py --sweep configs/sweep.yaml --dry-run
python main.py --sweep configs/sweep.yaml
```

Каждый джоб пишет свой каталог. Сводка sweep:

- `logs/sweeps/<name>_<timestamp>/summary.csv`
- `logs/sweeps/<name>_<timestamp>/ranking.json` — лучший джоб по val-метрике

Победителя потом прогоняют отдельно с TTA:

```shell
python main.py --downscale 1 --save-dir logs/final_best \
  --override dataset.augmentation.blur_prob=0.0 \
  --override dataset.augmentation.noise_prob=0.0 \
  --override train.epoch_fraction=0.25 \
  --override evaluation.tta=true
```

Ключи, которые обычно крутят в sweep:

- `train.epoch_fraction`, `train.backbone_lr`, `train.weight_decay`
- `model.dropout`, `model.drop_path_rate`
- `evaluation.tta`, `evaluation.early_stopping_patience`
