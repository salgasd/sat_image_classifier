# Satellite image classification

## Подготовка рабочего окружения для тренировки

```bash
make install_reqs_dev
make get_data
```

## Запуск тренировки

```bash
make train
```

Если нужно поменять базовый конфиг можно запустить так:

```
PYTHONPATH=. ./venv/bin/python3 src/train.py \
++model.net.model_name="efficientnet_b0" \
++data.batch_size=512 ++data.num_workers=11 \
++experiment_name="test"
```

## История экспериментов

В качестве трекера экспериментов использовался ClearML, для инициализации его нужно прописать в терминале

```bash
clearml-init
```

Лучшая модель в ClearML - https://app.clear.ml/projects/4a3437e1a278419287d282f41173fc08/experiments/7ba31d2ad65248aa9ba744769d5f9f14/output/execution

## Подготовка окружения для инференса

```bash
make install_reqs_infer
dvc pull
```

Пример скрипта для инференса есть тут: `src/infer.py`
