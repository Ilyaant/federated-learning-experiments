# Эксперименты по федеративному обучению
В этом репозитории содержатся эксперименты с датасетом CIFAR-100 и федеративным обучением
## Установка зависимостей
### Напрямую
**Требуется Python 3.12.10**

Для скачивания репозитория выполнить:
```shell
git clone https://github.com/Ilyaant/federated-learning-experiments.git
```
Для установки зависимостей выполнить
```shell
pip install -r requirements.txt
```
### С использованием Docker
Выполнить сборку образа:
```shell
docker build -t fl-experiments-image .
```
Запустить контейнер:
```shell
docker run -itd --name fl-experiments --gpus all fl-experiments-image /bin/bash
```
Внутрь контейнера клонировать репозиторий и выполнить установку зависимостей (см. предыдущий пункт).
## Запуск экспериментов
Для запуска эксперимента **с подходом data sharing** выполнить:
```shell
python __main__.py --shared_ratio=0.1
```
Для запуска эксперимента **без использования data sharing** выполнить:
```shell
python __main__.py --shared_ratio=0.0
```
Для запуска эксперимента со своими параметрами создать соответствующий конфиг в папке `configs` и выполнить:
```shell
python __main__.py --config_path=configs/<кастомный конфиг>.yaml
```
