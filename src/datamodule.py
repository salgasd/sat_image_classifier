import logging
import os
from typing import Optional

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from src.augmentations import get_transforms
from src.dataset import SatDataset
from src.dataset_splitter import stratify_shuffle_split_subsets


class SatDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
        train_size: float,
        pin_memory: bool,
        image_width: int,
        image_height: int,
    ):
        super().__init__()
        self._data_path = data_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_size = train_size
        self._pin_memory = pin_memory
        self._image_width = image_width
        self._image_height = image_height

        self._images_path = os.path.join(data_path, "train-jpg")
        self._train_transforms = get_transforms(
            width=self._image_width,
            height=self._image_height,
            preprocessing=False,
        )
        self._test_transforms = get_transforms(
            width=self._image_width,
            height=self._image_height,
            augmentations=False,
        )

    def prepare_data(self) -> None:
        if os.path.isfile(os.path.join(self._data_path, "df_train.csv")):
            logging.info("Using saved train/test/val splits")
            return

        df = pd.read_csv(os.path.join(self._data_path, "train_classes.csv"))
        logging.info("Original dataset has %s rows", len(df))
        df = df.drop_duplicates()
        logging.info("Dataset after removing dublicates has %s rows", len(df))

        df_train, df_val, df_test = stratify_shuffle_split_subsets(df, self._train_size)
        logging.info("Train size: %s", len(df_train))
        logging.info("Val size: %s", len(df_val))
        logging.info("Test size: %s", len(df_test))
        df_train.to_csv(
            os.path.join(self._data_path, "df_train.csv"),
            index=False,
        )
        df_val.to_csv(
            os.path.join(self._data_path, "df_val.csv"),
            index=False,
        )
        df_test.to_csv(
            os.path.join(self._data_path, "df_test.csv"),
            index=False,
        )
        logging.info("Datasets saved")

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit":
            df_train = pd.read_csv(os.path.join(self._data_path, "df_train.csv"))
            df_val = pd.read_csv(os.path.join(self._data_path, "df_val.csv"))
            self._train_dataset = SatDataset(
                df_train,
                image_folder=self._images_path,
                transforms=self._train_transforms,
            )
            self._val_dataset = SatDataset(
                df_val,
                image_folder=self._images_path,
                transforms=self._test_transforms,
            )
        elif stage == "test":
            df_test = pd.read_csv(os.path.join(self._data_path, "df_test.csv"))
            self._test_dataset = SatDataset(
                df_test,
                image_folder=self._images_path,
                transforms=self._test_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=self._pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self._batch_size,
            num_workers=1,
            shuffle=False,
            pin_memory=self._pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self._batch_size,
            num_workers=1,
            shuffle=False,
            pin_memory=self._pin_memory,
            drop_last=False,
        )
