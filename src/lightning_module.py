from typing import Any

import lightning as L
import torch
from torchmetrics import MeanMetric, MetricCollection


class SatModule(L.LightningModule):
    def __init__(  # type: ignore
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss,
        metrics: MetricCollection,
    ):
        super().__init__()
        self.model = net
        self._optimizer = optimizer(self.model.parameters())
        self._scheduler = scheduler(self._optimizer)
        self._loss = loss
        self.metrics = metrics
        self._val_metrics = metrics.clone(prefix="val_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self._val_metrics.reset()

    def training_step(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        loss = self._loss(logits, targets)
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: list[torch.Tensor]) -> None:
        images, targets = batch
        logits = self(images)
        loss = self._loss(logits, targets)
        preds = torch.sigmoid(logits)
        self._val_metrics(preds, targets)
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self._val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: list[torch.Tensor]) -> None:
        images, targets = batch
        logits = self(images)
        loss = self._loss(logits, targets)
        preds = torch.sigmoid(logits)
        self._test_metrics(preds, targets)
        self.test_loss(loss)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self._test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self._val_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": {
                "scheduler": self._scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
