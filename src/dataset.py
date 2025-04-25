import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class SatDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_folder: str,
        transforms: albu.BaseCompose | None = None,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.transforms = transforms
        self.image_paths = (self.image_folder + "/" + df['image_name'] + ".jpg").values
        self.labels = np.array(
            df.drop(columns=['image_name', 'tags']),
            dtype='float32',
        )
        self.classes = tuple(df.columns.drop(['image_name', 'tags']))

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, NDArray[np.float32]]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label

    def __len__(self) -> int:
        return len(self.image_paths)
