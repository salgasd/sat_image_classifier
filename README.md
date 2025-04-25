# Satellite image classification

Image classification pet project using Pytorch Lightning and Hydra

## Setting up the environment for training

```bash
make install_reqs_dev
make get_data
```

## Training the Model

```bash
make train
```

You can modify the base config:

```
PYTHONPATH=. ./venv/bin/python3 src/train.py \
++model.net.model_name="efficientnet_b0" \
++data.batch_size=512 ++data.num_workers=11 \
++experiment_name="test"
```

## Experiment History

ClearML was used as the experiment tracker. To initialize it, run the following command in the terminal:

```bash
clearml-init
```

## Setting up the environment for inference

```bash
make install_reqs_infer
dvc pull
```

An example inference script can be found here: `src/infer.py`
