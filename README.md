# Эксперименты по федеративному обучению
В этом репозитории содержатся эксперименты с датасетом CIFAR-100 и федеративным обучением
## Установка зависимостей
Для скачивания репозитория выполнить:
```shell
git clone https://github.com/Ilyaant/federated-learning-experiments.git
```
Для установки зависимостей выполнить
```shell
pip install -r requirements.txt
```
**Требуется Python 3.12.10**
## Запуск экспериментов
Для запуска эксперимента **с подходом data sharing** выполнить:
```shell
python __main__.py --config_path=configs/config_data_sharing.yaml
```
Для запуска эксперимента **без использования data sharing** выполнить:
```shell
python __main__.py --config_path=configs/config_0.yaml
```
Для запуска эксперимента со своими параметрами создать соответствующий конфиг в папке `configs` и выполнить:
```shell
python __main__.py --config_path=configs/<кастомный конфиг>.yaml
```
