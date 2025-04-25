import os

import hydra
import lightning as L
import torch
from clearml import Task
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    exp_path = os.path.join(cfg.paths.exp_path, cfg.experiment_name)
    os.makedirs(exp_path, exist_ok=True)

    if cfg.log_to_clearml:
        task = Task.init(  # noqa: F841
            project_name=cfg.project_name,
            task_name=cfg.experiment_name,
            auto_connect_frameworks=True,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_path,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_loss:.6f}",
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", mode="min", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = L.Trainer(
        max_epochs=cfg.n_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callbacks,
        precision=cfg.precision,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(
        ckpt_path=checkpoint_callback.best_model_path,
        datamodule=datamodule,
    )
    script = model.to_torchscript()
    torch.jit.save(script, os.path.join(exp_path, "model_jit.pt"))


if __name__ == "__main__":
    main()
